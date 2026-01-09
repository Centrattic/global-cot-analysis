/** @type {import('next').NextConfig} */
const nextConfig = {
    async headers() {
        const headers = []

        // Only add ngrok headers in development
        if (process.env.NODE_ENV === 'development') {
            headers.push({
                source: '/(.*)',
                headers: [
                    {
                        key: 'ngrok-skip-browser-warning',
                        value: 'true',
                    },
                ],
            })
        }

        return headers
    },
    async rewrites() {
        return [
            {
                source: '/data/:path*',
                destination: '/api/data/:path*',
            },
        ]
    },
    // Ensure proper handling of static files
    trailingSlash: false,
}

module.exports = nextConfig
