from __future__ import print_function, unicode_literals, absolute_import, division

from .care_standard import CARE
from ..internals import nets


class MultiplaneCARE(CARE):

    def _build(self):
        return nets.common_uxnet(
            n_dim=self.config.n_dim,
            prob_out=self.config.probabilistic,
            residual=self.config.unet_residual,
            n_depth=self.config.unet_n_depth,
            kern_size=self.config.unet_kern_size,
            n_first=self.config.unet_n_first,
            last_activation=self.config.unet_last_activation,
            shared_idx=self.config.shared_idx,
        )(self.config.unet_input_shape)
