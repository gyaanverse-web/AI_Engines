<script lang="ts">
	import LaTexRenderer from '$lib/LaTexRenderer.svelte';

	type QuestionType = 'theory' | 'numerical';

	type AnalysisStep = {
		stepId: string;
		text: string;
		step_status: 'right' | 'wrong' | 'unknown' | 'incomplete';
		description: string;
		topic: string;
		step_understanding: string;
		step_weight: number;
	};

	type QuestionItem = {
		id: string;
		topic: string;
		type: QuestionType;
		question: string;
	};

	type AnalysisApiResponse = {
		response: AnalysisStep[];
	};

	const questions: QuestionItem[] = [
		{
			id: 'Q1',
			topic: 'Science Class 9 - Work and Energy',
			type: 'numerical',
			question:
				'What is the work to be done to increase the velocity of a car from $30\\,\\text{km h}^{-1}$ to $60\\,\\text{km h}^{-1}$ if the mass of the car is $1500\\,\\text{kg}$?'
		}
	];

	let currentQuestionIndex = $state(0);
	let currentImageDataUrl = $state('');
	let currentImageName = $state('');
	let savedImageDataUrls = $state<string[]>(Array(questions.length).fill(''));
	let savedImageNames = $state<string[]>(Array(questions.length).fill(''));
	let isSubmitting = $state(false);
	let isAnalyzing = $state(false);
	let isFinished = $state(false);
	let error = $state<string | null>(null);
	let analysisData = $state<AnalysisApiResponse | null>(null);

	const currentQuestion = $derived(questions[currentQuestionIndex]);
	const progressLabel = $derived(`${currentQuestionIndex + 1} / ${questions.length}`);

	function loadCurrentSavedAnswer() {
		currentImageDataUrl = savedImageDataUrls[currentQuestionIndex] || '';
		currentImageName = savedImageNames[currentQuestionIndex] || '';
	}

	function saveCurrentAnswer() {
		savedImageDataUrls[currentQuestionIndex] = currentImageDataUrl;
		savedImageNames[currentQuestionIndex] = currentImageName;
	}

	function hasCurrentAnswer() {
		return Boolean(currentImageDataUrl);
	}

	async function handleImageChange(event: Event) {
		error = null;
		const target = event.target as HTMLInputElement;
		const file = target.files?.[0];
		if (!file) return;

		const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
		if (!allowedTypes.includes(file.type)) {
			error = 'Please upload PNG, JPG, JPEG or WEBP image.';
			target.value = '';
			return;
		}

		if (file.size > 10 * 1024 * 1024) {
			error = 'Image size should be less than 10MB.';
			target.value = '';
			return;
		}

		const dataUrl = await readFileAsDataUrl(file);
		currentImageDataUrl = dataUrl;
		currentImageName = file.name;
		saveCurrentAnswer();
	}

	function clearCurrentImage() {
		currentImageDataUrl = '';
		currentImageName = '';
		saveCurrentAnswer();
	}

	function goToNextQuestion() {
		error = null;
		if (!hasCurrentAnswer()) {
			error = 'Please upload answer image before moving to next question.';
			return;
		}

		saveCurrentAnswer();
		if (currentQuestionIndex < questions.length - 1) {
			currentQuestionIndex += 1;
			loadCurrentSavedAnswer();
		}
	}

	async function analyzeAnswer() {
		error = null;
		if (!hasCurrentAnswer()) {
			error = 'Please upload an answer image before running analysis.';
			return;
		}

		saveCurrentAnswer();
		isSubmitting = true;
		isAnalyzing = true;

		try {
			const payload = {
				solution_url: currentImageDataUrl,
				question: currentQuestion.question,
				top_k: 5
			};

			const response = await fetch('/api/analyze-image', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(payload)
			});

			if (!response.ok) {
				const body = await response.json().catch(() => ({}));
				throw new Error(body.error || 'Unable to analyze test right now.');
			}

			analysisData = (await response.json()) as AnalysisApiResponse;
			isFinished = true;
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to analyze image.';
		} finally {
			isSubmitting = false;
			isAnalyzing = false;
		}
	}

	function restartTest() {
		currentQuestionIndex = 0;
		currentImageDataUrl = '';
		currentImageName = '';
		savedImageDataUrls = Array(questions.length).fill('');
		savedImageNames = Array(questions.length).fill('');
		isSubmitting = false;
		isAnalyzing = false;
		isFinished = false;
		error = null;
		analysisData = null;
	}

	function readFileAsDataUrl(file: File): Promise<string> {
		return new Promise((resolve, reject) => {
			const reader = new FileReader();
			reader.onload = () => resolve(String(reader.result || ''));
			reader.onerror = () => reject(new Error('Failed to read image file.'));
			reader.readAsDataURL(file);
		});
	}

	function toMathRenderText(value: string): string {
		const cleaned = (value || '').trim();
		if (!cleaned) return '';
		return cleaned.includes('$') ? cleaned : `$${cleaned}$`;
	}

	function getSummary(steps: AnalysisStep[]) {
		return steps.reduce(
			(acc, step) => {
				acc.total += 1;
				acc[step.step_status] += 1;
				acc.scoreSum += step.step_status === 'right' ? 1 : step.step_status === 'incomplete' ? 0.5 : step.step_status === 'unknown' ? 0.3 : 0;
				return acc;
			},
			{
				total: 0,
				right: 0,
				wrong: 0,
				incomplete: 0,
				unknown: 0,
				scoreSum: 0
			}
		);
	}

	const analysisSummary = $derived(
		analysisData ? getSummary(analysisData.response) : null
	);
</script>

<div class="page">
	<div class="ambient one"></div>
	<div class="ambient two"></div>

	<main class="card">
		<header class="header">
			<p class="kicker">GyanVerse Analysis</p>
			<h1>Science Answer Review</h1>
			<p class="subtitle">Upload a solution image to analyze the extracted steps.</p>
		</header>

		{#if error}
			<div class="error">{error}</div>
		{/if}

		{#if !isFinished}
			<section class="question-panel">
				<div class="meta-row">
					<span class="pill">Question {progressLabel}</span>
					<span class="pill muted">{currentQuestion.topic}</span>
					<span class="pill type">{currentQuestion.type}</span>
				</div>

				<div class="question-text">
					<LaTexRenderer text={currentQuestion.question} />
				</div>

				<label for="student-image">Upload Answer Image</label>
				<input
					id="student-image"
					type="file"
					accept="image/png,image/jpeg,image/jpg,image/webp"
					onchange={handleImageChange}
					disabled={isSubmitting || isAnalyzing}
				/>

				{#if currentImageDataUrl}
					<div class="preview-card">
						<div class="preview-header">
							<span>{currentImageName}</span>
							<button class="tiny" type="button" onclick={clearCurrentImage}>Remove</button>
						</div>
						<img src={currentImageDataUrl} alt="Student answer preview" />
					</div>
				{/if}

				<div class="actions">
					{#if currentQuestionIndex < questions.length - 1}
						<button class="primary" onclick={goToNextQuestion} disabled={isSubmitting || isAnalyzing}>
							Submit Answer Image & Next
						</button>
					{:else}
						<button class="primary" onclick={analyzeAnswer} disabled={isSubmitting || isAnalyzing}>
							Analyze Image
						</button>
					{/if}
				</div>

				{#if isAnalyzing}
					<div class="analyzing-box">
						<div class="dot-loader">
							<span></span><span></span><span></span>
						</div>
						<p>Your test is being analyzed, we will display result shortly.</p>
					</div>
				{/if}
			</section>
		{:else if analysisData}
			<section class="report-panel">
				<div class="summary">
					<h2>Analysis Result</h2>
					<p>
						Score: <strong>{analysisSummary?.total ? Math.round((analysisSummary.scoreSum / analysisSummary.total) * 100) : 0}%</strong>
					</p>
					<p>
						Right: {analysisSummary?.right ?? 0} | Wrong: {analysisSummary?.wrong ?? 0} | Incomplete:
						{analysisSummary?.incomplete ?? 0} | Unknown: {analysisSummary?.unknown ?? 0}
					</p>
				</div>

				<div class="report-list">
					{#each analysisData.response as item}
						<article class="report-item">
							<div class="report-head">
								<span class="status">{item.step_status.toUpperCase()}</span>
								<span class="topic">{item.topic}</span>
							</div>
							<h3><LaTexRenderer text={toMathRenderText(item.text)} /></h3>
							<p><strong>Issue:</strong> {item.description}</p>
							{#if item.step_understanding}
								<p><strong>Step understanding:</strong> {item.step_understanding}</p>
							{/if}
							<p><strong>Weight:</strong> {item.step_weight}</p>
						</article>
					{/each}
				</div>

				<button class="secondary" onclick={restartTest}>Analyze Another Image</button>
			</section>
		{/if}
	</main>
</div>

<style>
	:global(*) {
		box-sizing: border-box;
	}

	:global(body) {
		margin: 0;
		font-family: 'Source Sans 3', 'Segoe UI', Tahoma, sans-serif;
		background: radial-gradient(circle at 10% 10%, #f5f9ff, transparent 45%),
			radial-gradient(circle at 90% 20%, #e9f7ef, transparent 40%),
			linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
		min-height: 100vh;
		color: #1c2738;
	}

	.page {
		min-height: 100vh;
		padding: 32px 16px;
		position: relative;
		overflow: hidden;
	}

	.ambient {
		position: absolute;
		border-radius: 999px;
		filter: blur(30px);
		opacity: 0.55;
		z-index: 0;
	}

	.ambient.one {
		width: 220px;
		height: 220px;
		background: #d7e9ff;
		top: -40px;
		left: -40px;
	}

	.ambient.two {
		width: 280px;
		height: 280px;
		background: #d6f3df;
		bottom: -120px;
		right: -60px;
	}

	.card {
		position: relative;
		z-index: 1;
		max-width: 860px;
		margin: 0 auto;
		background: rgba(255, 255, 255, 0.9);
		backdrop-filter: blur(8px);
		border: 1px solid #dbe4ee;
		border-radius: 18px;
		padding: 28px;
		box-shadow: 0 16px 50px rgba(17, 38, 74, 0.08);
	}

	.header {
		margin-bottom: 20px;
	}

	.kicker {
		font-size: 13px;
		text-transform: uppercase;
		letter-spacing: 1.2px;
		font-weight: 700;
		color: #3669a8;
		margin: 0 0 10px;
	}

	h1 {
		margin: 0;
		font-size: 34px;
		line-height: 1.1;
	}

	.subtitle {
		margin-top: 8px;
		color: #526278;
	}

	.question-panel,
	.report-panel {
		padding-top: 6px;
	}

	.meta-row {
		display: flex;
		gap: 10px;
		flex-wrap: wrap;
		margin-bottom: 12px;
	}

	.pill {
		display: inline-flex;
		padding: 6px 12px;
		border-radius: 999px;
		font-size: 12px;
		font-weight: 700;
		background: #edf4ff;
		color: #214c7b;
	}

	.pill.muted {
		background: #ecf7ef;
		color: #24593e;
	}

	.pill.type {
		background: #fff4e5;
		color: #925d13;
	}

	h2 {
		margin: 0 0 8px;
		font-size: 24px;
	}

	.question-text {
		font-size: 24px;
		font-weight: 700;
		line-height: 1.35;
		margin: 0 0 8px;
	}

	label {
		display: block;
		margin-bottom: 8px;
		font-weight: 700;
	}

	input[type='file'] {
		width: 100%;
		border: 1px solid #bfd0e2;
		border-radius: 12px;
		padding: 10px;
		background: #fff;
	}

	.preview-card {
		margin-top: 12px;
		border: 1px solid #d6e2ef;
		border-radius: 12px;
		padding: 10px;
		background: #fbfdff;
	}

	.preview-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 8px;
		font-size: 13px;
		font-weight: 700;
		color: #3d5269;
	}

	.preview-card img {
		width: 100%;
		max-height: 360px;
		object-fit: contain;
		border-radius: 10px;
		background: #fff;
		border: 1px solid #e6edf5;
	}

	.actions {
		margin-top: 16px;
		display: flex;
		justify-content: flex-end;
	}

	button {
		border: none;
		border-radius: 10px;
		padding: 12px 18px;
		font-size: 14px;
		font-weight: 700;
		cursor: pointer;
	}

	button:disabled {
		opacity: 0.65;
		cursor: not-allowed;
	}

	.tiny {
		padding: 6px 10px;
		font-size: 12px;
		background: #f0f4f9;
		color: #34495f;
	}

	.primary {
		background: #1f6fb2;
		color: #fff;
	}

	.secondary {
		background: #e8eef5;
		color: #27415d;
		margin-top: 14px;
	}

	.analyzing-box {
		margin-top: 16px;
		padding: 14px;
		border-radius: 12px;
		background: #eef6ff;
		border: 1px solid #c9ddf3;
		display: flex;
		align-items: center;
		gap: 12px;
	}

	.dot-loader {
		display: inline-flex;
		gap: 4px;
	}

	.dot-loader span {
		width: 8px;
		height: 8px;
		border-radius: 999px;
		background: #2f6fab;
		animation: pulse 1.1s infinite ease-in-out;
	}

	.dot-loader span:nth-child(2) {
		animation-delay: 0.15s;
	}

	.dot-loader span:nth-child(3) {
		animation-delay: 0.3s;
	}

	@keyframes pulse {
		0%,
		100% {
			transform: translateY(0);
			opacity: 0.5;
		}
		50% {
			transform: translateY(-4px);
			opacity: 1;
		}
	}

	.error {
		padding: 10px 12px;
		margin: 0 0 14px;
		border-radius: 10px;
		background: #fff1f1;
		border: 1px solid #ffcaca;
		color: #992c2c;
	}

	.summary {
		margin-bottom: 14px;
		padding: 14px;
		border-radius: 12px;
		background: #f3f8ff;
		border: 1px solid #d5e3f4;
	}

	.summary p {
		margin: 6px 0 0;
		color: #425870;
	}

	.report-list {
		display: grid;
		gap: 12px;
	}

	.report-item {
		border-radius: 12px;
		padding: 14px;
		border: 1px solid #d9e4ef;
		background: #fff;
	}

	.report-head {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		margin-bottom: 10px;
		font-size: 12px;
		font-weight: 700;
	}

	.report-item h3 {
		margin: 0 0 8px;
		font-size: 17px;
	}

	.report-item p {
		margin: 6px 0;
		line-height: 1.5;
		color: #33485f;
	}

	@media (max-width: 700px) {
		.card {
			padding: 18px;
		}

		h1 {
			font-size: 28px;
		}

		h2 {
			font-size: 21px;
		}
	}
</style>
