# global
import startai


class PreNorm(startai.Module):
    def __init__(
        self, dim, fn, key_dim=None, value_dim=None, eps=1e-05, device=None, v=None
    ):
        self._attention = fn
        self._norm = startai.LayerNorm(dim, eps=eps, device=device)
        self._norm_key = (
            startai.LayerNorm(key_dim, eps=eps, device=device)
            if startai.exists(key_dim)
            else None
        )
        self._norm_value = (
            startai.LayerNorm(value_dim, eps=eps, device=device)
            if startai.exists(value_dim)
            else None
        )
        startai.Module.__init__(self, v=v, device=device)

    def _forward(self, *args, **kwargs):
        args = list(args)
        args[0] = self._norm(args[0])
        if startai.exists(self._norm_key):
            args[1] = self._norm_key(args[1])
        if startai.exists(self._norm_value):
            args[2] = self._norm_value(args[2])
        return self._attention(*args, **kwargs)


class FeedForward(startai.Module):
    def __init__(self, dim, dropout=0.0, device=None, v=None):
        self._net = startai.Sequential(
            startai.Linear(dim, dim, device=device),
            startai.GELU(),
            startai.Linear(dim, dim, device=device),
            startai.Dropout(dropout),
            device=device,
        )
        startai.Module.__init__(self, v=v)

    def _forward(self, x):
        return self._net(x)


def _perceiver_jax_weights_mapping(old_key, new_key):
    new_mapping = new_key
    if "proj_weights" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "a b -> b a"}
    elif "output/w" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "a b -> b a"}
    return new_mapping
