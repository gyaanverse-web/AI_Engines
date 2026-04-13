import { env } from '$env/dynamic/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from '@sveltejs/kit';
import http from 'node:http';
import https from 'node:https';
import { URL } from 'node:url';

type AnalyzeImageRequest = {
	solution_url?: string;
	image_source?: string;
	answerImage?: string;
	question?: string;
	top_k?: number;
	collection_name?: string;
};

export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = (await request.json()) as AnalyzeImageRequest;
		const solutionUrl = (body.solution_url || body.image_source || body.answerImage || '').trim();
		const question = (body.question || '').trim();
		const collectionName = (body.collection_name || env.ANALYSIS_COLLECTION_NAME || '').trim();
		const topK = Number.isFinite(body.top_k) ? body.top_k : 5;

		if (!solutionUrl) {
			return json({ error: 'solution_url is required.' }, { status: 400 });
		}

		const backendBaseUrl = env.BACKEND_BASE_URL || 'http://127.0.0.1:5000';
		const backendPayload: Record<string, unknown> = {
			image_source: solutionUrl,
			question,
			top_k: topK
		};

		if (collectionName) {
			backendPayload.collection_name = collectionName;
		}

		const backendResponse = await postJson(backendBaseUrl, '/get_analysis', backendPayload);
		console.log(`[api/analyze-image] Backend status: ${backendResponse.status}`);

		if (backendResponse.status < 200 || backendResponse.status >= 300) {
			console.error('[api/analyze-image] Backend error body:', backendResponse.body);
			let parsedBody: { error?: string } = {};
			try {
				parsedBody = JSON.parse(backendResponse.body) as { error?: string };
			} catch {
				// Keep the raw body for logging; return a generic error to the client.
			}
			return json(
				{ error: parsedBody.error || 'Failed to analyze image.' },
				{ status: backendResponse.status }
			);
		}

		return json(JSON.parse(backendResponse.body));
	} catch (error) {
		console.error('analyze-image error', error);
		return json({ error: 'Failed to analyze image.' }, { status: 500 });
	}
};

function postJson(baseUrl: string, path: string, payload: unknown): Promise<{ status: number; body: string }> {
	return new Promise((resolve, reject) => {
		const targetUrl = new URL(path, baseUrl);
		const body = JSON.stringify(payload);
		const transport = targetUrl.protocol === 'https:' ? https : http;

		const request = transport.request(
			{
				method: 'POST',
				hostname: targetUrl.hostname,
				port: targetUrl.port || undefined,
				path: `${targetUrl.pathname}${targetUrl.search}`,
				headers: {
					'Content-Type': 'application/json',
					'Content-Length': Buffer.byteLength(body)
				}
			},
			(response) => {
				let responseBody = '';
				response.setEncoding('utf8');
				response.on('data', (chunk) => {
					responseBody += chunk;
				});
				response.on('end', () => {
					resolve({ status: response.statusCode || 500, body: responseBody });
				});
			}
		);

		request.on('error', reject);
		request.write(body);
		request.end();
	});
}
