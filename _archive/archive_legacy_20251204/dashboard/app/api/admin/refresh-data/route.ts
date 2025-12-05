import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const { source } = await request.json();

    // TODO: Trigger actual data refresh
    // For now, return success
    const sources = ['databento', 'fred', 'usda', 'eia', 'noaa', 'all'];
    
    if (!source || !sources.includes(source)) {
      return NextResponse.json(
        { error: 'Invalid source. Must be one of: ' + sources.join(', ') },
        { status: 400 }
      );
    }

    // Mock refresh response
    const response = {
      success: true,
      source,
      message: `Refresh triggered for ${source}`,
      estimatedTime: '5-10 minutes',
      timestamp: new Date().toISOString()
    };

    return NextResponse.json(response);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to trigger refresh' },
      { status: 500 }
    );
  }
}

