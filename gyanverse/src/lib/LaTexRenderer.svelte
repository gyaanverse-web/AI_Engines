<script lang="ts">
	import katex from 'katex';
	import 'katex/dist/katex.min.css';

	let { text = '' } = $props();

	let renderedHtml = $state('');

	// Parse and render LaTeX
	function renderLatex(input: string) {
		const parts = [];
		let lastIndex = 0;
		
		// Regex to match both $$ (display) and $ (inline) math
		// Match $$ first, then $
		const regex = /(\$\$[\s\S]*?\$\$|\$[^\$\n]+\$)/g;
		let match;

		while ((match = regex.exec(input)) !== null) {
			// Add text before the match
			if (match.index > lastIndex) {
				parts.push({
					type: 'text',
					content: input.slice(lastIndex, match.index)
				});
			}

			const mathText = match[0];
			const isDisplay = mathText.startsWith('$$');
			const latex = isDisplay 
				? mathText.slice(2, -2) 
				: mathText.slice(1, -1);

			try {
				const html = katex.renderToString(latex, {
					throwOnError: false,
					displayMode: isDisplay,
					output: 'html'
				});

				parts.push({
					type: isDisplay ? 'display' : 'inline',
					content: html
				});
			} catch (e) {
				console.error('KaTeX error:', e);
				parts.push({
					type: 'text',
					content: mathText
				});
			}

			lastIndex = regex.lastIndex;
		}

		// Add remaining text
		if (lastIndex < input.length) {
			parts.push({
				type: 'text',
				content: input.slice(lastIndex)
			});
		}

		return parts;
	}

	$effect(() => {
		const parts = renderLatex(text);
		renderedHtml = parts
			.map(part => {
				if (part.type === 'text') {
					return `<span>${part.content.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</span>`;
				} else if (part.type === 'display') {
					return `<div class="math-display">${part.content}</div>`;
				} else {
					return `<span class="math-inline">${part.content}</span>`;
				}
			})
			.join('');
	});
</script>

<div class="latex-content">
	{@html renderedHtml}
</div>

<style>
	:global(.katex) {
		font-size: 1em;
	}

	:global(.math-inline .katex) {
		display: inline;
	}

	.latex-content :global(.math-display) {
		display: flex;
		justify-content: center;
		margin: 12px 0;
		padding: 8px;
		background: #f5f5f5;
		border-radius: 6px;
	}

	.latex-content :global(.math-display .katex) {
		font-size: 1.1em;
	}

	.latex-content {
		line-height: 1.6;
		word-wrap: break-word;
	}
</style>
