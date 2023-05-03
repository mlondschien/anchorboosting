import numpy as np


class AnchorMixin:
    def _proj(self, anchor, f):
        """Project f onto the subspace spanned by anchor.

        Parameters
        ----------
        anchor: np.ndarray of dimension (n, d_anchor).
            The anchor matrix. If d_anchor = 1 and entries are integer, assumed to be
            categories.
        f: np.ndarray of dimension (n, d_f) or (n,).
            The vector to project.

        Returns
        -------
        np.ndarray of dimension (n, d_f) or (n,)
            Projection of f onto the subspace spanned by anchor. Same dimension as f.
        """
        if anchor.shape[1] == 1 and anchor.dtype == int:
            out = np.zeros(f.shape)
            for unique_value in np.unique(anchor):
                mask = (anchor == unique_value).flatten()
                if len(f.shape) == 1:
                    out[mask] = f[mask].mean()
                else:
                    out[mask, :] = f[mask, :].mean(axis=0)
            return out

        anchor = anchor - anchor.mean(axis=0)
        f_mean = f.mean(axis=0)

        return np.dot(anchor, np.linalg.lstsq(anchor, f, rcond=None)[0]) + f_mean

    def _proj_matrix(self, a):
        """Projection matrix onto the subspace spanned by a."""
        assert a.shape[1] < a.shape[0]
        return np.dot(np.dot(a, np.linalg.inv(a.T @ a)), a.T)
