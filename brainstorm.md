
Goal: We want to find SAE latents that forms an FSM and see it's structure across various tasks

I found two latents fot html that seem to activate serially:

```python
html_features = {
    "open tags": {
        "layer": 12,
        "idx" : 14778
    },
    "anchor tags": {
        "layer": 12,
        "idx" : 18916
    }
}
```

Question: 
1. Can we find latents like this using attribution?
    1. What does like this mean:
       latents that are responsible for activating the next token
    2. Since I am assuming each token independently, I want to find the contribution of each latent upto token t towards the next token. 