/**
 * Trigger.dev Webhook Handler for Next.js
 * 
 * Handles incoming webhooks from Trigger.dev for job status updates.
 * Based on: https://trigger.dev/docs/guides/frameworks/nextjs-webhooks
 */

import { NextRequest, NextResponse } from "next/server";
import { verifySignature } from "@trigger.dev/sdk/v3";

export async function POST(request: NextRequest) {
  try {
    // Verify webhook signature
    const signature = request.headers.get("x-trigger-signature");
    const body = await request.text();

    if (!signature) {
      return NextResponse.json(
        { error: "Missing signature" },
        { status: 401 }
      );
    }

    const isValid = verifySignature({
      signature,
      body,
      secret: process.env.TRIGGER_SECRET_KEY!,
    });

    if (!isValid) {
      return NextResponse.json(
        { error: "Invalid signature" },
        { status: 401 }
      );
    }

    // Parse webhook payload
    const payload = JSON.parse(body);

    console.log("[Trigger Webhook] Received:", {
      event: payload.event,
      jobId: payload.job?.id,
      status: payload.job?.status,
    });

    // Handle different webhook events
    switch (payload.event) {
      case "job.completed":
        await handleJobCompleted(payload);
        break;

      case "job.failed":
        await handleJobFailed(payload);
        break;

      case "job.started":
        await handleJobStarted(payload);
        break;

      default:
        console.log(`[Trigger Webhook] Unhandled event: ${payload.event}`);
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("[Trigger Webhook] Error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

/**
 * Handle job completion
 */
async function handleJobCompleted(payload: any) {
  const { job } = payload;

  console.log(`[Trigger Webhook] Job completed: ${job.id}`);
  console.log(`  - Duration: ${job.duration}ms`);
  console.log(`  - Output:`, job.output);

  // TODO: Store job results in MotherDuck logs table
  // TODO: Send notification if critical job
  // TODO: Trigger downstream jobs if needed
}

/**
 * Handle job failure
 */
async function handleJobFailed(payload: any) {
  const { job } = payload;

  console.error(`[Trigger Webhook] Job failed: ${job.id}`);
  console.error(`  - Error:`, job.error);
  console.error(`  - Attempts: ${job.attempts}`);

  // TODO: Send alert to Slack/email
  // TODO: Log to error tracking system
  // TODO: Retry if appropriate
}

/**
 * Handle job start
 */
async function handleJobStarted(payload: any) {
  const { job } = payload;

  console.log(`[Trigger Webhook] Job started: ${job.id}`);
  console.log(`  - Task: ${job.task}`);
  console.log(`  - Payload:`, job.payload);

  // TODO: Update dashboard with job status
}

