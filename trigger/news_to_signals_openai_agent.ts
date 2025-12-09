/**
 * News-to-Signals OpenAI Agent - Trigger.dev Job
 * 
 * Uses OpenAI Agents SDK with guardrails to process news articles into trading signals.
 * Analyzes sentiment, impact magnitude, time horizon, and ZL directional bias.
 * 
 * Based on:
 * - https://trigger.dev/docs/guides/example-projects/openai-agent-sdk-guardrails
 * - https://trigger.dev/docs/guides/examples/vercel-ai-sdk
 */

import { task } from "@trigger.dev/sdk/v3";
import { openai } from "@ai-sdk/openai";
import { generateObject } from "ai";
import { z } from "zod";
import { MotherDuckClient } from "../src/shared/motherduck_client";

// Schema for news signal output
const NewsSignalSchema = z.object({
  article_id: z.string(),
  
  // Sentiment Analysis
  zl_sentiment: z.enum(["BULLISH_ZL", "BEARISH_ZL", "NEUTRAL"]),
  sentiment_confidence: z.number().min(0).max(1),
  sentiment_reasoning: z.string(),
  
  // Impact Assessment
  impact_magnitude: z.enum(["HIGH", "MEDIUM", "LOW"]),
  impact_reasoning: z.string(),
  
  // Time Horizon
  horizon: z.enum(["FLASH", "TACTICAL", "STRUCTURAL"]),
  horizon_reasoning: z.string(),
  
  // Thematic Classification
  theme_primary: z.enum([
    "SUPPLY_WEATHER",
    "DEMAND_BIOFUELS",
    "DEMAND_CHINA",
    "TRADE_GEO",
    "MACRO_FX",
    "LOGISTICS",
    "POSITIONING",
    "IDIOSYNCRATIC",
  ]),
  
  // Policy Axis (if applicable)
  policy_axis: z.enum([
    "TRADE_CHINA",
    "TRADE_TARIFFS",
    "BIOFUELS_RFS",
    "BIOFUELS_LCFS",
    "BIOFUELS_45Z",
    "IMMIGRATION_H2A",
    "NONE",
  ]).optional(),
  
  // Key Entities Mentioned
  entities: z.object({
    commodities: z.array(z.string()),
    countries: z.array(z.string()),
    companies: z.array(z.string()),
    policies: z.array(z.string()),
  }),
  
  // Quantitative Signals (if extractable)
  price_targets: z.array(z.object({
    commodity: z.string(),
    target: z.number(),
    timeframe: z.string(),
  })).optional(),
});

type NewsSignal = z.infer<typeof NewsSignalSchema>;

/**
 * Process a single article with OpenAI Agent
 */
async function processArticleWithAgent(
  article: { article_id: string; headline: string; content: string; source: string }
): Promise<NewsSignal> {
  const prompt = `
You are a quantitative trading analyst specializing in soybean oil (ZL) futures.

Analyze this news article and extract trading signals:

**Source:** ${article.source}
**Headline:** ${article.headline}
**Content:** ${article.content}

Your task:
1. Determine ZL sentiment (BULLISH_ZL, BEARISH_ZL, NEUTRAL)
2. Assess impact magnitude (HIGH, MEDIUM, LOW)
3. Classify time horizon (FLASH=immediate, TACTICAL=days-weeks, STRUCTURAL=months-years)
4. Identify primary theme (SUPPLY_WEATHER, DEMAND_BIOFUELS, DEMAND_CHINA, TRADE_GEO, MACRO_FX, etc.)
5. Extract key entities (commodities, countries, companies, policies)
6. If mentioned, extract price targets

**Context for ZL (Soybean Oil):**
- BULLISH drivers: China demand ↑, biofuel mandates ↑, crude oil ↑, crush margins ↑, USD ↓
- BEARISH drivers: China demand ↓, biofuel mandates ↓, crude oil ↓, crush margins ↓, USD ↑
- Related commodities: ZS (soybeans), ZM (soybean meal), CL (crude oil), HG (copper as China proxy)

Provide detailed reasoning for each classification.
`;

  const result = await generateObject({
    model: openai("gpt-4o"),
    schema: NewsSignalSchema,
    prompt,
    temperature: 0.3, // Lower temperature for more consistent analysis
  });

  return result.object;
}

/**
 * Main Trigger.dev task: News-to-Signals with OpenAI Agent
 */
export const newsToSignalsAgent = task({
  id: "news-to-signals-openai-agent",
  retry: {
    maxAttempts: 2,
    factor: 2,
    minTimeoutInMs: 5000,
    maxTimeoutInMs: 60000,
  },
  run: async (payload: { batchSize?: number; lookbackHours?: number }, { ctx }) => {
    const batchSize = payload.batchSize || 50;
    const lookbackHours = payload.lookbackHours || 24;

    console.log(`[News-to-Signals] Processing articles from last ${lookbackHours} hours...`);

    // Fetch unprocessed articles from MotherDuck
    const motherduck = new MotherDuckClient();

    const articles = await motherduck.query<{
      article_id: string;
      headline: string;
      content: string;
      source: string;
    }>(`
      SELECT article_id, headline, content, source
      FROM raw.bucket_news
      WHERE created_at >= NOW() - INTERVAL '${lookbackHours} hours'
        AND article_id NOT IN (SELECT article_id FROM features.news_signals)
      LIMIT ${batchSize}
    `);

    console.log(`[News-to-Signals] Found ${articles.length} unprocessed articles`);

    if (articles.length === 0) {
      return {
        success: true,
        articlesProcessed: 0,
        message: "No new articles to process",
        timestamp: new Date().toISOString(),
      };
    }

    // Process articles with OpenAI Agent
    const signals: NewsSignal[] = [];
    const errors: string[] = [];

    for (const article of articles) {
      try {
        console.log(`[News-to-Signals] Processing: ${article.headline.slice(0, 60)}...`);
        const signal = await processArticleWithAgent(article);
        signals.push(signal);
      } catch (error) {
        console.error(`[News-to-Signals] Error processing ${article.article_id}:`, error);
        errors.push(article.article_id);
      }
    }

    // Store signals in MotherDuck
    if (signals.length > 0) {
      await motherduck.insertBatch("features.news_signals", signals);
      console.log(`[News-to-Signals] Stored ${signals.length} signals`);
    }

    return {
      success: true,
      articlesProcessed: signals.length,
      errors: errors.length,
      errorArticles: errors,
      timestamp: new Date().toISOString(),
    };
  },
});

