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
        return np.dot(anchor, np.linalg.lstsq(anchor, f, rcond=None)[0])

    def _proj_matrix(self, a):
        """Projection matrix onto the subspace spanned by a."""
        assert a.shape[1] < a.shape[0]
        return np.dot(np.dot(a, np.linalg.inv(a.T @ a)), a.T)
