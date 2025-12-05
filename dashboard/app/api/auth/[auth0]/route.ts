import { NextRequest } from 'next/server';

// Auth0 v4 SDK requires environment variables to be set:
// AUTH0_SECRET, AUTH0_BASE_URL, AUTH0_ISSUER_BASE_URL, AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET

export async function GET(req: NextRequest) {
    // Placeholder for Auth0 v4 implementation
    // Configure AUTH0_* environment variables and use @auth0/nextjs-auth0 v4 API
    return new Response('Auth not configured', { status: 501 });
}

export async function POST(req: NextRequest) {
    return new Response('Auth not configured', { status: 501 });
}
