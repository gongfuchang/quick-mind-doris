import { NextRequest, NextResponse } from "next/server";

export const config = {
  matcher: ["/api/query-server"],
};

export function middleware(req: NextRequest) {

  return NextResponse.next({
    request: {
      headers: req.headers,
    },
  });
}
