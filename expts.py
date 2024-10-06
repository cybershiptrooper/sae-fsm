# %%
import torch
from sae_lens import SAE
import transformer_lens as tl
import yaml
import sae_lens
from typing import List

torch.set_grad_enabled(False)

# %%
model = tl.HookedTransformer.from_pretrained("gpt2-small")

# %%
device = model.cfg.device

# %%
html_prompt = """<html>
<head>
    <title>My Website</title>
</head>
<body>
    <h>Welcome to My Website</h1>
    <p>This is a paragraph.</p>
    <div class="content">
        <h2>Here is a heading</h2>
        <p>This is another paragraph.</p>
    </div>
</html>
"""
html_prompt

# %%
html_tokens = model.tokenizer(html_prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
bos_token_id = model.tokenizer.bos_token_id
if bos_token_id is not None:
    html_tokens = torch.cat([torch.tensor([[bos_token_id]]).to(device), html_tokens], dim=1)
model.to_str_tokens(html_tokens[0])

# %%
str_tokens = model.to_str_tokens(html_tokens[0])
tokens_we_care_about = [i for i in range(len(str_tokens)) if '<' in str_tokens[i] and "<|endoftext|>" != str_tokens[i]]
tokens_we_care_about

# %%
tokens_we_care_about = tokens_we_care_about[1:-1]
tokens_we_care_about

# %%
logits = model(html_tokens)

# %%
import neel_utils as nutils
nutils.show_df(nutils.create_vocab_df(logits[0, tokens_we_care_about[0]], make_probs=True).head(15))

# %% [markdown]
# ### Load SAEs

# %%
with open("/usr/local/lib/python3.10/dist-packages/sae_lens/pretrained_saes.yaml", "r") as file:
    pretrained_saes = yaml.safe_load(file)
print(pretrained_saes.keys())


RELEASE = "gpt2-small-res-jb"

saes = {}
hook_points = []
for layer in range(model.cfg.n_layers + 1):
    sae_id = f"blocks.{layer}.hook_resid_pre"
    if layer == 12:
        sae_id = f"blocks.{layer-1}.hook_resid_post"

    hook_points.append(sae_id)

    saes[layer] = sae_lens.SAE.from_pretrained(
        release=RELEASE,
        sae_id=sae_id,
        device="cuda",
    )[0].to(torch.float32)
hook_points

# %% [markdown]
# # Feature attribution

# %%
def get_cache_fwd_and_bwd(model, tokens, metric, layers):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    for layer in layers:
        model.add_hook(hook_points[layer], forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    for layer in layers:
        model.add_hook(hook_points[layer], backward_cache_hook, "bwd")
    torch.set_grad_enabled(True)
    logits = model(tokens.clone())
    value = metric(logits)
    value.backward()
    torch.set_grad_enabled(False)
    model.reset_hooks()
    return (
        value.item(),
        tl.ActivationCache(cache, model),
        tl.ActivationCache(grad_cache, model),
    )

# %%
def metric(logits: torch.Tensor, idxs_we_care_about: List[int] = tokens_we_care_about, labels: torch.Tensor = html_tokens[:, 1:]):
    """Next token prediction"""
    logits_to_use = logits[:, idxs_we_care_about]
    labels_to_use = labels[:, idxs_we_care_about]
    # print(model.to_str_tokens(labels_to_use))
    log_probs = torch.log_softmax(logits_to_use, dim=-1)
    # print(log_probs.shape)
    # print(labels_to_use[..., None].shape)
    log_probs_at_labels = log_probs.gather(dim=-1, index=labels_to_use[..., None])
    # import math
    # with torch.no_grad():
    #     print(math.e**log_probs_at_labels) 
    return log_probs_at_labels.mean()

# %%
loss, fwd_cache, bwd_cache = get_cache_fwd_and_bwd(
    model, html_tokens, metric, list(range(model.cfg.n_layers + 1))
)
loss

# %%
ALL_LAYERS = list(range(model.cfg.n_layers + 1))
resids = torch.stack([fwd_cache[hook_points[layer]].squeeze(0) for layer in ALL_LAYERS])
grad_resids = torch.stack(
    [bwd_cache[hook_points[layer]].squeeze(0) for layer in ALL_LAYERS]
)

width = 16
sae_acts_list = []
sae_attrs_list = []
for c, layer in enumerate(ALL_LAYERS):
    recons_resids, sae_cache = saes[layer].run_with_cache(resids[c].float())
    sae_acts = sae_cache[f"hook_sae_acts_post"]
    sae_attrs = (grad_resids[c].float() @ saes[layer].W_dec.T) * sae_acts
    sae_acts_list.append(sae_acts)
    sae_attrs_list.append(sae_attrs)
sae_acts = torch.stack(sae_acts_list)
sae_attrs = torch.stack(sae_attrs_list)

# %%
sae_attrs.shape # Shape: layers x token pos x latent

# %%
from neel_plotly import *

line(
    sae_attrs.sum(-1),
    x=nutils.process_tokens_index(html_tokens),
    color=[str(l) for l in ALL_LAYERS],
    title="SAE Attribution Scores (sum)",
    # labels={"x": "Tokens", "y": "Attribution Score"},
)
line(
    sae_attrs.max(-1).values,
    x=nutils.process_tokens_index(html_tokens),
    color=[str(l) for l in ALL_LAYERS],
    title="SAE Attribution Scores (max)",
    # labels={"x": "Tokens", "y": "Attribution Score"},
)
# line(
#     sae_attrs.min(-1).values,
#     x=nutils.process_tokens_index(html_tokens),
#     color=[str(l) for l in ALL_LAYERS],
#     title="SAE Attribution Scores (min)",
#     # labels={"x": "Tokens", "y": "Attribution Score"},
# )

# %%
sae_attrs_we_care_about = sae_attrs[:, tokens_we_care_about]
line(
    sae_attrs_we_care_about.sum(1),
    # x=nutils.process_tokens_index(generated_tokens[:56]),
    color=["Layer " + str(l) for l in ALL_LAYERS],
    title="SAE Attribution Scores (sum)",
    # labels={"x": "Tokens", "y": "Attribution Score"},
)
line(
    sae_attrs_we_care_about.max(1).values,
    # x=nutils.process_tokens_index(generated_tokens[:56]),
    color=["Layer " + str(l) for l in ALL_LAYERS],
    title="SAE Attribution Scores (max)",
    # labels={"x": "Tokens", "y": "Attribution Score"},
)

# %%
# line(
#     sae_attrs.sum(0),
#     # x=nutils.process_tokens_index(html_tokens),
#     color=[str(l) for l in ALL_LAYERS],
#     title="SAE Attribution Scores (sum)",
#     # labels={"x": "Tokens", "y": "Attribution Score"},
# )
# line(
#     sae_attrs.max(0).values,
#     # x=nutils.process_tokens_index(html_tokens),
#     color=[str(l) for l in ALL_LAYERS],
#     title="SAE Attribution Scores (max)",
#     # labels={"x": "Tokens", "y": "Attribution Score"},
# )


