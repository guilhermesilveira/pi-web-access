import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { activityMonitor } from "./activity.js";
import { getApiKey, API_BASE, DEFAULT_MODEL } from "./gemini-api.js";
import { isGeminiWebAvailable, queryWithCookies } from "./gemini-web.js";
import type { GoogleAccount } from "./chrome-cookies.js";
import { isPerplexityAvailable, searchWithPerplexity, type SearchResult, type SearchResponse, type SearchOptions } from "./perplexity.js";

export type SearchProvider = "auto" | "perplexity" | "gemini";

const CONFIG_PATH = join(homedir(), ".pi", "web-search.json");

interface SearchConfig {
	searchProvider: SearchProvider;
	googleAccount?: string;
}

let cachedSearchConfig: SearchConfig | null = null;

function getSearchConfig(): SearchConfig {
	if (cachedSearchConfig) return cachedSearchConfig;
	try {
		if (existsSync(CONFIG_PATH)) {
			const raw = JSON.parse(readFileSync(CONFIG_PATH, "utf-8"));
			cachedSearchConfig = {
				searchProvider: raw.searchProvider ?? "auto",
				googleAccount: raw.googleAccount,
			};
			return cachedSearchConfig;
		}
	} catch {}
	cachedSearchConfig = { searchProvider: "auto" };
	return cachedSearchConfig;
}

/** Clear cached config so changes to web-search.json take effect immediately. */
export function clearSearchConfigCache(): void {
	cachedSearchConfig = null;
}

/** Get the configured Google account email (if any). */
export function getConfiguredGoogleAccount(): string | undefined {
	return getSearchConfig().googleAccount;
}

/**
 * Resolve a configured Google account email to an account index.
 * Returns undefined (use default account) if no match or no config.
 */
export function resolveAccountIndex(accounts: GoogleAccount[], configuredEmail?: string): number | undefined {
	if (!configuredEmail || accounts.length <= 1) return undefined;
	const match = accounts.find((a) => a.email.toLowerCase() === configuredEmail.toLowerCase());
	if (!match) return undefined;
	// Index 0 = primary, no need for /u/0/ prefix
	return match.index === 0 ? undefined : match.index;
}

export interface FullSearchOptions extends SearchOptions {
	provider?: SearchProvider;
}

export async function search(query: string, options: FullSearchOptions = {}): Promise<SearchResponse> {
	const config = getSearchConfig();
	const provider = options.provider ?? config.searchProvider;

	if (provider === "perplexity") {
		return searchWithPerplexity(query, options);
	}

	if (provider === "gemini") {
		const result = await searchWithGeminiApi(query, options)
			?? await searchWithGeminiWeb(query, options);
		if (result) return result;
		throw new Error(
			"Gemini search unavailable. Either:\n" +
			"  1. Set GEMINI_API_KEY in ~/.pi/web-search.json\n" +
			"  2. Sign into gemini.google.com in Chrome"
		);
	}

	if (isPerplexityAvailable()) {
		return searchWithPerplexity(query, options);
	}

	const geminiResult = await searchWithGeminiApi(query, options)
		?? await searchWithGeminiWeb(query, options);
	if (geminiResult) return geminiResult;

	throw new Error(
		"No search provider available. Either:\n" +
		"  1. Set perplexityApiKey in ~/.pi/web-search.json\n" +
		"  2. Set GEMINI_API_KEY in ~/.pi/web-search.json\n" +
		"  3. Sign into gemini.google.com in Chrome"
	);
}

async function searchWithGeminiApi(query: string, options: SearchOptions = {}): Promise<SearchResponse | null> {
	const apiKey = getApiKey();
	if (!apiKey) return null;

	const activityId = activityMonitor.logStart({ type: "api", query });

	try {
		const model = DEFAULT_MODEL;
		const body = {
			contents: [{ parts: [{ text: query }] }],
			tools: [{ google_search: {} }],
		};

		const res = await fetch(`${API_BASE}/models/${model}:generateContent?key=${apiKey}`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(body),
			signal: AbortSignal.any([
				AbortSignal.timeout(60000),
				...(options.signal ? [options.signal] : []),
			]),
		});

		if (!res.ok) {
			const errorText = await res.text();
			throw new Error(`Gemini API error ${res.status}: ${errorText.slice(0, 300)}`);
		}

		const data = await res.json() as GeminiSearchResponse;
		activityMonitor.logComplete(activityId, res.status);

		const answer = data.candidates?.[0]?.content?.parts
			?.map(p => p.text).filter(Boolean).join("\n") ?? "";

		const metadata = data.candidates?.[0]?.groundingMetadata;
		const results = await resolveGroundingChunks(metadata?.groundingChunks, options.signal);

		if (!answer && results.length === 0) return null;
		return { answer, results };
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		if (message.toLowerCase().includes("abort")) {
			activityMonitor.logComplete(activityId, 0);
		} else {
			activityMonitor.logError(activityId, message);
		}
		return null;
	}
}

async function searchWithGeminiWeb(query: string, options: SearchOptions = {}): Promise<SearchResponse | null> {
	const availability = await isGeminiWebAvailable();
	if (!availability) return null;

	const config = getSearchConfig();
	const accountIndex = resolveAccountIndex(availability.accounts, config.googleAccount);

	const prompt = buildSearchPrompt(query, options);
	const activityId = activityMonitor.logStart({ type: "api", query });

	try {
		const text = await queryWithCookies(prompt, availability.cookies, {
			model: "gemini-3-flash-preview",
			signal: options.signal,
			timeoutMs: 60000,
			accountIndex,
		});

		activityMonitor.logComplete(activityId, 200);

		const results = extractSourceUrls(text);
		return { answer: text, results };
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		if (message.toLowerCase().includes("abort")) {
			activityMonitor.logComplete(activityId, 0);
		} else {
			activityMonitor.logError(activityId, message);
		}
		return null;
	}
}

function buildSearchPrompt(query: string, options: SearchOptions): string {
	let prompt = `Search the web and answer the following question. Include source URLs for your claims.\nFormat your response as:\n1. A direct answer to the question\n2. Cited sources as markdown links\n\nQuestion: ${query}`;

	if (options.recencyFilter) {
		const labels: Record<string, string> = {
			day: "past 24 hours",
			week: "past week",
			month: "past month",
			year: "past year",
		};
		prompt += `\n\nOnly include results from the ${labels[options.recencyFilter]}.`;
	}

	if (options.domainFilter?.length) {
		const includes = options.domainFilter.filter(d => !d.startsWith("-"));
		const excludes = options.domainFilter.filter(d => d.startsWith("-")).map(d => d.slice(1));
		if (includes.length) prompt += `\n\nOnly cite sources from: ${includes.join(", ")}`;
		if (excludes.length) prompt += `\n\nDo not cite sources from: ${excludes.join(", ")}`;
	}

	return prompt;
}

function extractSourceUrls(markdown: string): SearchResult[] {
	const results: SearchResult[] = [];
	const seen = new Set<string>();
	const linkRegex = /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g;
	for (const match of markdown.matchAll(linkRegex)) {
		const url = match[2];
		if (seen.has(url)) continue;
		seen.add(url);
		results.push({ title: match[1], url, snippet: "" });
	}
	return results;
}

async function resolveGroundingChunks(
	chunks: GroundingChunk[] | undefined,
	signal?: AbortSignal,
): Promise<SearchResult[]> {
	if (!chunks?.length) return [];

	const results: SearchResult[] = [];
	for (const chunk of chunks) {
		if (!chunk.web) continue;
		const title = chunk.web.title || "";
		let url = chunk.web.uri || "";

		if (url.includes("vertexaisearch.cloud.google.com/grounding-api-redirect")) {
			const resolved = await resolveRedirect(url, signal);
			if (resolved) url = resolved;
		}

		if (url) results.push({ title, url, snippet: "" });
	}
	return results;
}

async function resolveRedirect(proxyUrl: string, signal?: AbortSignal): Promise<string | null> {
	try {
		const res = await fetch(proxyUrl, {
			method: "HEAD",
			redirect: "manual",
			signal: AbortSignal.any([
				AbortSignal.timeout(5000),
				...(signal ? [signal] : []),
			]),
		});
		return res.headers.get("location") || null;
	} catch {
		return null;
	}
}

interface GeminiSearchResponse {
	candidates?: Array<{
		content?: { parts?: Array<{ text?: string }> };
		groundingMetadata?: {
			webSearchQueries?: string[];
			groundingChunks?: GroundingChunk[];
			groundingSupports?: Array<{
				segment?: { startIndex?: number; endIndex?: number; text?: string };
				groundingChunkIndices?: number[];
			}>;
		};
	}>;
}

interface GroundingChunk {
	web?: { uri?: string; title?: string };
}
