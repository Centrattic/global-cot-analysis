#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    echo "NGROK_AUTHTOKEN=your_ngrok_token_here" > .env
    echo "Please edit .env file and add your ngrok token, then run this script again."
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check if NGROK_AUTHTOKEN is set
if [ -z "$NGROK_AUTHTOKEN" ] || [ "$NGROK_AUTHTOKEN" = "your_ngrok_token_here" ]; then
    echo "Please set your NGROK_AUTHTOKEN in the .env file"
    exit 1
fi

echo "Starting Next.js development server with ngrok tunnel..."
npm run dev
