/**
 * ProFarmer Anchor Browser Scraper - Trigger.dev Job
 * 
 * Uses Anchor browser automation for authenticated scraping of ProFarmer.
 * Handles JavaScript-rendered content and login flows.
 * 
 * Based on: https://trigger.dev/docs/guides/example-projects/anchor-browser-web-scraper
 */

import { task } from "@trigger.dev/sdk/v3";
import { MotherDuckClient } from "../../../src/shared/motherduck_client";

interface ProFarmerArticle {
  article_id: string;
  headline: string;
  content: string;
  author: string;
  source: string;
  source_trust_score: number;
  published_at: string;
  url: string;
  edition_type: string;
  bucket_name: string;
}

/**
 * Anchor browser scraping with authentication
 */
async function scrapeProFarmerWithAnchor(
  section: { name: string; url: string; edition_type: string }
): Promise<ProFarmerArticle[]> {
  const { Anchor } = await import("@ancorai/sdk");
  
  const anchor = new Anchor({
    apiKey: process.env.ANCHOR_API_KEY!,
  });

  try {
    // Step 1: Navigate to ProFarmer login
    await anchor.goto("https://www.profarmer.com/login");

    // Step 2: Fill login form
    await anchor.fill("#username", process.env.PROFARMER_USERNAME!);
    await anchor.fill("#password", process.env.PROFARMER_PASSWORD!);
    await anchor.click("button[type='submit']");

    // Step 3: Wait for login to complete
    await anchor.waitForNavigation();

    // Step 4: Navigate to section
    await anchor.goto(`https://www.profarmer.com${section.url}`);

    // Step 5: Extract articles using AI-powered selectors
    const articles = await anchor.extract({
      schema: {
        articles: {
          type: "array",
          items: {
            type: "object",
            properties: {
              headline: { type: "string", selector: "h2, h3" },
              url: { type: "string", selector: "a[href]", attribute: "href" },
              author: { type: "string", selector: ".author, .byline" },
              date: { type: "string", selector: "time, .date" },
              summary: { type: "string", selector: "p, .excerpt" },
            },
          },
        },
      },
    });

    // Step 6: Scrape full article content (first 500 words)
    const fullArticles: ProFarmerArticle[] = [];

    for (const article of articles.articles.slice(0, 10)) {
      try {
        await anchor.goto(article.url);

        const content = await anchor.extract({
          schema: {
            body: { type: "string", selector: ".article-body, article" },
          },
        });

        // Limit to 500 words
        const words = content.body.split(/\s+/);
        const limitedContent = words.slice(0, 500).join(" ") + (words.length > 500 ? "..." : "");

        fullArticles.push({
          article_id: Buffer.from(article.url).toString("base64").slice(0, 32),
          headline: article.headline,
          content: limitedContent,
          author: article.author || "ProFarmer",
          source: "ProFarmer",
          source_trust_score: 0.95,
          published_at: article.date || new Date().toISOString(),
          url: article.url,
          edition_type: section.edition_type,
          bucket_name: "profarmer_anchor",
        });
      } catch (error) {
        console.error(`[Anchor] Error scraping article ${article.url}:`, error);
      }
    }

    return fullArticles;
  } finally {
    await anchor.close();
  }
}

/**
 * Main Trigger.dev task: ProFarmer Anchor Scraper
 */
export const profarmerAnchorScraper = task({
  id: "profarmer-anchor-scraper",
  retry: {
    maxAttempts: 3,
    factor: 2,
    minTimeoutInMs: 5000,
    maxTimeoutInMs: 30000,
  },
  run: async (payload: { sections?: string[] }, { ctx }) => {
    const SECTIONS = [
      { name: "First Thing Today", url: "/news/first-thing-today", edition_type: "pre_open" },
      { name: "Ahead of the Open", url: "/news/ahead-of-the-open", edition_type: "pre_open" },
      { name: "After the Bell", url: "/news/after-the-bell", edition_type: "post_close" },
      { name: "Agriculture News", url: "/news/agriculture-news", edition_type: "intraday" },
      { name: "Newsletters", url: "/newsletters", edition_type: "newsletter" },
    ];

    const sectionsToScrape = payload.sections
      ? SECTIONS.filter((s) => payload.sections!.includes(s.name))
      : SECTIONS;

    console.log(`[ProFarmer Anchor] Scraping ${sectionsToScrape.length} sections...`);

    const allArticles: ProFarmerArticle[] = [];

    for (const section of sectionsToScrape) {
      try {
        console.log(`[ProFarmer Anchor] Scraping ${section.name}...`);
        const articles = await scrapeProFarmerWithAnchor(section);
        allArticles.push(...articles);
        console.log(`[ProFarmer Anchor] Found ${articles.length} articles in ${section.name}`);
      } catch (error) {
        console.error(`[ProFarmer Anchor] Error scraping ${section.name}:`, error);
      }
    }

    // Load to MotherDuck
    if (allArticles.length > 0) {
      const motherduck = new MotherDuckClient();
      await motherduck.insertBatch("raw.scrapecreators_news_buckets", allArticles);
      console.log(`[ProFarmer Anchor] Loaded ${allArticles.length} articles to MotherDuck`);
    }

    return {
      success: true,
      articlesScraped: allArticles.length,
      sections: sectionsToScrape.map((s) => s.name),
      timestamp: new Date().toISOString(),
    };
  },
});

