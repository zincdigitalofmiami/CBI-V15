/**
 * Trigger.dev Webhook Handler Stub
 * 
 * Placeholder for Trigger.dev webhook integration.
 * Full implementation requires @trigger.dev/sdk/v3 to be installed.
 */

import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  // Stub implementation - Trigger.dev SDK not installed in dashboard
  console.log("[Trigger Webhook] Received webhook (stub mode)");
  
  try {
    const body = await request.text();
    const payload = JSON.parse(body);
    
    console.log("[Trigger Webhook] Event:", payload.event);
    console.log("[Trigger Webhook] Job ID:", payload.job?.id);
    
    return NextResponse.json({ 
      success: true, 
      message: "Webhook received (stub mode)" 
    });
  } catch (error) {
    console.error("[Trigger Webhook] Error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({ 
    status: "ok", 
    message: "Trigger webhook endpoint (stub mode)" 
  });
}
