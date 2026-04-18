window.FRACTAL_VISUAL_PAGES = [
    { key: "gpt", href: "./gpt.html", label: "GPT / Transformer" },
    { key: "hybrid", href: "./hybrid.html", label: "Hybrid / Jamba" },
    { key: "p20", href: "./p20.html", label: "P20 / Recurrent Control" },
    { key: "fractal-v1", href: "./fractal-v1.html", label: "Fractal v1" },
    { key: "fractal-v2", href: "./fractal-v2.html", label: "Fractal v2" },
];

window.FRACTAL_VISUAL_CONFIGS = {
    gpt: {
        key: "gpt",
        sceneType: "gpt",
        tag: "Transformer Baseline",
        title: "GPT-Like LLM",
        subtitle:
            "A stacked transformer shown as a 3D token lattice: residual stream in the middle, attention bridges across tokens, and a KV cache that grows during inference.",
        accent: "#7ce9ff",
        accent2: "#8f78ff",
        success: "#52f0b7",
        warning: "#ffd56e",
        danger: "#ff6b8f",
        sceneHint: "Drag to rotate. Scroll to zoom. Use the mode buttons to switch runtime views.",
        structure: [
            "Tokenizer and embeddings turn text into a token ribbon.",
            "A deep stack of transformer blocks processes all visible tokens.",
            "Multi-head attention compares token positions inside each attention layer.",
            "Residual streams and MLPs fuse head outputs before the LM head predicts the next token.",
        ],
        legend: [
            { color: "#7ce9ff", label: "Token stream and KV cache state" },
            { color: "#8f78ff", label: "Attention layers and head bridges" },
            { color: "#52f0b7", label: "Residual / fusion flow" },
            { color: "#ff6b8f", label: "Backward pressure during training" },
        ],
        facts: [
            {
                title: "Memory substrate",
                body: "Token-wise memory. Every token keeps a visible address in the sequence and can be revisited explicitly through attention.",
            },
            {
                title: "Why it works",
                body: "Query-specific retrieval is sharp. Exact copy, long-range comparison, and induction all come naturally from dense token access.",
            },
            {
                title: "Main cost",
                body: "Attention compares many token pairs. Prefill gets expensive fast as context grows, even when FlashAttention makes it practical.",
            },
        ],
        modes: {
            training: {
                label: "Training",
                short: "Full forward + backward",
                caption:
                    "All tokens participate, all layers are active, and gradients travel through every attention edge and residual stream.",
                insight:
                    "This is why transformers are reliable to optimize: the retrieval mechanism is explicit. It is also why training cost is large.",
                metrics: {
                    focus: "Full-stack forward + backward through dense attention",
                    memory: "Residual activations + attention weights + optimizer state",
                    compute: "High, especially with long context",
                },
            },
            prefill: {
                label: "Prefill",
                short: "Context ingestion",
                caption:
                    "The prompt is pushed through the whole stack so the KV cache can be populated before generation starts.",
                insight:
                    "Prefill is where transformers pay the biggest context-length tax. The whole visible prompt participates together.",
                metrics: {
                    focus: "Populate KV cache across all visible tokens",
                    memory: "KV grows with sequence length and layers",
                    compute: "Attention over the whole prompt window",
                },
            },
            decode: {
                label: "Decode",
                short: "One-token generation",
                caption:
                    "A fresh query token reads the cached keys and values from prior tokens while the rest of the context stays frozen in memory.",
                insight:
                    "Decode is cheaper than prefill because cached history is reused, but the model still depends on token-addressable memory.",
                metrics: {
                    focus: "One new query attends to prior cached history",
                    memory: "Large persistent KV cache",
                    compute: "Per-token attention over prior context",
                },
            },
            synthesis: {
                label: "Synthesis",
                short: "Head fusion to logits",
                caption:
                    "Different heads gather different evidence, MLPs remix it, the residual stream preserves continuity, and the LM head turns the fused state into next-token logits.",
                insight:
                    "Attention is not the output layer. It is the selective retrieval engine feeding the synthesis path.",
                metrics: {
                    focus: "Head outputs fuse into a shared residual stream",
                    memory: "Diverse head-level evidence",
                    compute: "Parallel heads + feed-forward fusion",
                },
            },
        },
    },
    hybrid: {
        key: "hybrid",
        sceneType: "hybrid",
        tag: "Hybrid Backbone",
        title: "Jamba / Nemotron-Style Hybrid",
        subtitle:
            "A 3D hybrid stack where attention slabs alternate with fast state-space or recurrent channels, preserving explicit retrieval only where it earns its cost.",
        accent: "#72f4cc",
        accent2: "#738dff",
        success: "#89ff8a",
        warning: "#ffd56e",
        danger: "#ff7e91",
        sceneHint: "The teal spine is the cheap state path. Purple slabs are the expensive but precise retrieval layers.",
        structure: [
            "Tokenizer and embeddings still feed a standard autoregressive path.",
            "Attention layers are sparse or intermittent rather than everywhere.",
            "State-space or recurrent channels carry most of the sequence processing.",
            "The model keeps explicit attention for retrieval-critical moments instead of making it the whole substrate.",
        ],
        legend: [
            { color: "#72f4cc", label: "Selective state / recurrent path" },
            { color: "#738dff", label: "Attention or retrieval-heavy layers" },
            { color: "#89ff8a", label: "Fusion and gating path" },
            { color: "#ff7e91", label: "Backward pressure" },
        ],
        facts: [
            {
                title: "Memory substrate",
                body: "Mixed. Cheap recurrent or SSM state handles most flow; attention remains available when exact retrieval matters.",
            },
            {
                title: "Why it works",
                body: "The hybrid path reduces the amount of quadratic work without asking a single compressed state to do everything.",
            },
            {
                title: "Tradeoff",
                body: "Architecturally more complex than pure transformers or pure recurrent models, but much more plausible than betting everything on one memory mechanism.",
            },
        ],
        modes: {
            training: {
                label: "Training",
                short: "Alternating paths",
                caption:
                    "Gradients flow through both the cheap state path and the more selective attention slabs, so the model can decide where explicit retrieval is worth paying for.",
                insight:
                    "This is why hybrids keep winning the practical tradeoff: they reduce attention frequency instead of trying to abolish retrieval entirely.",
                metrics: {
                    focus: "State path plus periodic attention backprop",
                    memory: "Smaller than dense transformer, larger than pure recurrence",
                    compute: "Balanced: cheap layers dominate, costly layers are selective",
                },
            },
            prefill: {
                label: "Prefill",
                short: "Context compression",
                caption:
                    "Most layers stream the prompt through a cheap state path while selected attention slabs inject precise long-range context.",
                insight:
                    "Prefill stays more tractable because not every layer runs dense attention over the whole prompt.",
                metrics: {
                    focus: "State accumulation with periodic retrieval refresh",
                    memory: "Reduced attention surfaces",
                    compute: "Lower than pure transformer at long context",
                },
            },
            decode: {
                label: "Decode",
                short: "Cheap token stepping",
                caption:
                    "Generation mostly rides the state path and only taps the retrieval slabs when sharp token-addressable access is needed.",
                insight:
                    "This is the strongest deployment story for hybrids: fast decode without giving up all explicit memory.",
                metrics: {
                    focus: "State-driven decode with selective retrieval",
                    memory: "Hybrid persistent state plus smaller retrieval memory",
                    compute: "Closer to linear most of the time",
                },
            },
            synthesis: {
                label: "Synthesis",
                short: "Gated fusion",
                caption:
                    "State features and attention-derived evidence are fused by gates before the LM head predicts the next token.",
                insight:
                    "The model can use cheap continuous context most of the time and explicit retrieval when the decision boundary is sharp.",
                metrics: {
                    focus: "Gate state evidence against selective retrieved evidence",
                    memory: "Two complementary representations",
                    compute: "Selective fusion, not always-on global comparison",
                },
            },
        },
    },
    p20: {
        key: "p20",
        sceneType: "p20",
        tag: "Rotary Gated Recurrent Control",
        title: "P20 / Rotary Gated Recurrent Control",
        subtitle:
            "A 3D walkthrough of the model we tested: prelude token features feed one packed recurrent primitive, the primitive scans across the sequence, and its emitted control gates what enters a looped middle attention scaffold.",
        accent: "#78ecff",
        accent2: "#9a7bff",
        success: "#55f2b8",
        warning: "#ffd166",
        danger: "#ff6f91",
        sceneHint:
            "Follow the bottom token ribbon into the packed projection, then watch the central recurrent state drive the looped attention chamber.",
        structure: [
            "Prelude attention blocks turn raw tokens into residual-stream features.",
            "A packed input projection splits each token into update gate, rotary angle, candidate state, and output gate.",
            "A latent recurrent state is transformed, rotated, gated, and blended with the fresh candidate at each time step.",
            "The emitted recurrent output becomes a control signal for the looped middle attention scaffold, not a full attention replacement.",
        ],
        legend: [
            { color: "#78ecff", label: "Token/residual stream from the prelude" },
            { color: "#ffd166", label: "Packed control projections and gates" },
            { color: "#55f2b8", label: "Latent recurrent state and emitted control" },
            { color: "#9a7bff", label: "Looped middle attention chamber" },
            { color: "#ff6f91", label: "Backward pressure and proof-boundary caveats" },
        ],
        facts: [
            {
                title: "What P20 means here",
                body: "P20 is the project shorthand for this rotary gated recurrent state update primitive. Publicly, the important idea is the recurrence: transformed carry, rotary angle, candidate blend, and gated emitted output.",
            },
            {
                title: "Why the page looks different from GPT",
                body: "A transformer layer shows explicit token-to-token retrieval. This primitive shows a compressed latent state plus a control readout, so the visualization emphasizes the state update law and the injection seam.",
            },
            {
                title: "What the result currently supports",
                body: "The recurrent-control scaffold acts like an efficient depth substitute at tiny scale, but deeper attention still catches it. This page visualizes the mechanism, not a proof of broad transformer replacement.",
            },
        ],
        modes: {
            anatomy: {
                label: "Anatomy",
                short: "Packed projections + state law",
                caption:
                    "The primitive turns each prelude token feature into four streams: update gate, rotary angle, candidate state, and output gate. Those streams control one recurrent state update.",
                insight:
                    "The crucial design is not just recurrence. It is a typed recurrence with separate control projections and a distinct emitted output.",
                metrics: {
                    focus: "Packed in-projection to gated rotary recurrence",
                    memory: "One latent state vector plus emitted control",
                    compute: "Projection-heavy front end, cheap scan update",
                },
            },
            scan: {
                label: "Sequence Scan",
                short: "State flows token by token",
                caption:
                    "During scan, the state is updated causally from left to right. The previous state is transformed and rotated before it competes with the fresh candidate.",
                insight:
                    "This is the Mamba/GDN-adjacent part of the story: the model processes sequence history through a persistent latent state rather than a full KV table.",
                metrics: {
                    focus: "Causal step/scan equivalence",
                    memory: "Compressed recurrent state",
                    compute: "Linear in sequence length, kernel quality matters",
                },
            },
            loop: {
                label: "Loop Control",
                short: "Inject into middle depth",
                caption:
                    "The emitted recurrent output is normalized and converted into a value/gate pair that feeds the looped middle attention chamber.",
                insight:
                    "This is why the current result is an efficient-depth result: the primitive controls repeated computation rather than replacing every attention layer.",
                metrics: {
                    focus: "Control signal into looped middle blocks",
                    memory: "Loop state plus recurrent control",
                    compute: "Fewer physical layers, repeated middle compute",
                },
            },
            training: {
                label: "Training",
                short: "Shared recurrence under BPTT",
                caption:
                    "Gradients must pass through the looped scaffold and the shared recurrent scan. Stable gates and explicit projection contracts matter because the same primitive is reused many times.",
                insight:
                    "The visual trick from llm-viz is useful here: separate the forward objects from the backward pressure so we do not mistake an elegant diagram for a free training path.",
                metrics: {
                    focus: "Looped scaffold + recurrent scan gradients",
                    memory: "Activations for looped attention and scan",
                    compute: "Slower than the simplest looped control, faster than sufficiently deeper attention in the tested band",
                },
            },
            boundary: {
                label: "Proof Boundary",
                short: "What the data does and does not say",
                caption:
                    "The current H100 ladder says the recurrent-control path reaches the 14-layer attention boundary with fewer parameters, but a 16-layer attention model still wins on loss.",
                insight:
                    "This is the honest conclusion: promising enough for a grant-scale follow-up, not enough to claim a general transformer replacement.",
                metrics: {
                    focus: "Evidence boundary, not architecture hype",
                    memory: "Lower than deeper attention at comparable quality band",
                    compute: "Needs 30M-50M validation before stronger claims",
                },
            },
        },
    },
    "fractal-v1": {
        key: "fractal-v1",
        sceneType: "fractal-v1",
        tag: "What We Tried",
        title: "Fractal v1: Single-Root Recursive Kernel",
        subtitle:
            "The architecture we actually built: one recurrent root, inner recursive depth per token, a router, structured projections, and an LM head. This page highlights both the elegance and the training break point.",
        accent: "#67efff",
        accent2: "#52f0b7",
        success: "#9aff92",
        warning: "#ffd56e",
        danger: "#ff6b8f",
        sceneHint: "The central orb is the recurrent state. The red lattice in failure view is the unrolled training graph pressure, not a forward-memory diagram.",
        structure: [
            "A single recurrent state tries to carry the whole sequence memory.",
            "Each token triggers inner recursive depth instead of an explicit multiscale memory structure.",
            "A router can in principle early-exit, but it did not buy enough in the canary lane.",
            "Structured projections and the LM head made the system cleaner, but they did not solve the training formulation mismatch.",
        ],
        legend: [
            { color: "#67efff", label: "Token inflow and recurrent state" },
            { color: "#52f0b7", label: "Gated contractive update path" },
            { color: "#ffd56e", label: "Router and intended adaptive compute" },
            { color: "#ff6b8f", label: "Training break / backward pressure" },
        ],
        facts: [
            {
                title: "What was real",
                body: "The top p1 family had a genuine identity highway and could learn something locally. The structured projection and diagnostics work were worth doing.",
            },
            {
                title: "What fooled us",
                body: "Primitive tournaments validated the cell locally, not the full system. We over-inferred from local trainability to global training viability.",
            },
            {
                title: "Deeper issue",
                body: "Within one token, the inner loop behaves too much like repeated application of the same affine-like map. Full unrolled BPTT then pays for every reuse as if each step were distinct thought.",
            },
        ],
        modes: {
            structure: {
                label: "Structure",
                short: "What we built",
                caption:
                    "One recurrent root, three structured projections, inner recursive depth, router, and LM head. Clean in form, but narrow in memory geometry.",
                insight:
                    "This was a serious architecture, not a toy. The issue is not that it had no structure. The issue is what that structure asked one state to do.",
                metrics: {
                    focus: "Single-root recurrent memory",
                    memory: "Compressed state only",
                    compute: "Token loop × inner recursive depth",
                },
            },
            forward: {
                label: "Forward",
                short: "Why it looked promising",
                caption:
                    "The gated contractive update gives a smooth forward path. The identity highway stabilizes state propagation and makes the primitive look trainable at small scale.",
                insight:
                    "This is the signal the tournaments captured correctly: the cell itself is not nonsense.",
                metrics: {
                    focus: "State propagation with a residual gate",
                    memory: "One state vector carries the whole past",
                    compute: "Cheap-looking forward recurrence",
                },
            },
            backward: {
                label: "Backward",
                short: "Where pressure accumulated",
                caption:
                    "Backprop does not only care about stable state flow. It must accumulate parameter gradients across every recursive reuse and every token position in the unrolled graph.",
                insight:
                    "The training burden lives in the full shared-weight graph, not just in one isolated projection or one stable forward pass.",
                metrics: {
                    focus: "Shared-weight BPTT over sequence × depth",
                    memory: "Large implicit training graph",
                    compute: "Backward complexity dominates the elegant forward story",
                },
            },
            failure: {
                label: "Failure map",
                short: "Why we pivoted",
                caption:
                    "The canary debugging lane kept narrowing the seam, but the deeper issue is architectural: one compressed recurrent state plus inner recursion is an awkward substitute for explicit content-addressable memory.",
                insight:
                    "This is why v2 moves fractality into memory organization and sparse routing instead of forcing the primitive to become attention by itself.",
                metrics: {
                    focus: "Training formulation mismatch",
                    memory: "Too much burden on one state and one unrolled graph",
                    compute: "Fixing kernels did not fix the overall training shape",
                },
            },
        },
    },
    "fractal-v2": {
        key: "fractal-v2",
        sceneType: "fractal-v2",
        tag: "Successor Direction",
        title: "Fractal v2: Recursive Memory Kernel",
        subtitle:
            "The new direction: local recurrent processing inside causal leaf blocks, a dyadic summary tree above them, sparse routing heads, and exact leaf reads when copy-level fidelity matters.",
        accent: "#77f0ff",
        accent2: "#7a78ff",
        success: "#54f2ba",
        warning: "#ffd56e",
        danger: "#ff7f94",
        sceneHint: "Leaves sit on the floor, the memory tree rises above them, and routing heads descend only where the query needs sharper detail.",
        structure: [
            "Leaf blocks handle local causal processing.",
            "Completed leaves summarize upward into a dyadic tree.",
            "Multiple roots carry parallel state trajectories rather than collapsing everything into one root.",
            "Routing heads read sparsely from the tree and can descend to exact local leaf reads.",
        ],
        legend: [
            { color: "#77f0ff", label: "Leaf blocks and local trunks" },
            { color: "#7a78ff", label: "Dyadic tree and routing heads" },
            { color: "#54f2ba", label: "Root fusion and memory reads" },
            { color: "#ffd56e", label: "Teacher / supervision signal during training" },
        ],
        facts: [
            {
                title: "Primary bet",
                body: "Recurrence still does most of the work. The difference is that addressable memory is now explicit, sparse, multiscale, and causal.",
            },
            {
                title: "Why it is more plausible",
                body: "It preserves state as the dominant substrate while giving the model a clean answer to long-range retrieval and exact local copy.",
            },
            {
                title: "What must be proven",
                body: "Routing must specialize, tree summaries must stay useful, and exact leaf reads must help without collapsing back into dense attention everywhere.",
            },
        ],
        modes: {
            training: {
                label: "Training",
                short: "Teacher-guided sparse memory",
                caption:
                    "Leaf trunks, tree summaries, routing heads, and the LM head all train together. A teacher or auxiliary retrieval losses can supervise routing instead of hoping next-token loss invents it alone.",
                insight:
                    "v2 should not repeat v1's mistake of making the training objective discover all routing structure from scratch.",
                metrics: {
                    focus: "Local trunk + tree + sparse router + LM head",
                    memory: "Explicit bounded memory plus local recurrent state",
                    compute: "Aimed at sparse multiscale credit assignment",
                },
            },
            prefill: {
                label: "Prefill",
                short: "Build the tree as blocks complete",
                caption:
                    "Tokens fill the current leaf block, completed leaves summarize upward, and the tree grows incrementally instead of storing every token pairwise relationship everywhere.",
                insight:
                    "The extra log factor is spent on query-specific search, not on dense all-token comparison.",
                metrics: {
                    focus: "Incremental leaf completion and tree construction",
                    memory: "Leaf buffers + dyadic summaries",
                    compute: "Near-linear local work plus sparse multiscale updates",
                },
            },
            decode: {
                label: "Decode",
                short: "Update one leaf and a few ancestors",
                caption:
                    "A new token only mutates the active leaf block and the O(log n) summaries above it, then routing heads descend to the most relevant coarse or fine regions.",
                insight:
                    "This is the runtime story we actually want: no full KV sweep, no single-state memory bottleneck.",
                metrics: {
                    focus: "Current leaf update plus sparse ancestor refresh",
                    memory: "Persistent tree and bounded routing state",
                    compute: "Aimed at O(log n) update overhead per new token",
                },
            },
            synthesis: {
                label: "Synthesis",
                short: "Fuse roots and retrieved summaries",
                caption:
                    "Some routing heads stop at coarse summaries, others descend to leaves for exact reads, and all selected evidence is fused with the parallel root states before the LM head predicts the next token.",
                insight:
                    "Fractality now lives in multiscale memory and routing geometry, not in hoping one primitive formula becomes attention by itself.",
                metrics: {
                    focus: "Coarse-to-fine routing and fusion to logits",
                    memory: "Parallel roots plus exact local reads when needed",
                    compute: "Sparse query-specific synthesis rather than dense retrieval",
                },
            },
        },
    },
};
