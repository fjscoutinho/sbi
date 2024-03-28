# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.mnle import MixedDensityEstimator
from sbi.types import TorchTransform
from sbi.utils import mcmc_transform
from sbi.utils.sbiutils import match_theta_and_x_batch_shapes
from sbi.utils.torchutils import atleast_2d


def likelihood_estimator_based_potential(
    likelihood_estimator: nn.Module,
    prior: Distribution,
    x_o: Optional[Tensor],
    enable_transform: bool = True,
) -> Tuple[Callable, TorchTransform]:
    r"""Returns potential $\log(p(x_o|\theta)p(\theta))$ for likelihood-based methods.

    It also returns a transformation that can be used to transform the potential into
    unconstrained space.

    Args:
        likelihood_estimator: The neural network modelling the likelihood.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the likelihood.
        enable_transform: Whether to transform parameters to unconstrained space.
             When False, an identity transform will be returned for `theta_transform`.

    Returns:
        The potential function $p(x_o|\theta)p(\theta)$ and a transformation that maps
        to unconstrained space.
    """

    device = str(next(likelihood_estimator.parameters()).device)

    potential_fn = LikelihoodBasedPotential(
        likelihood_estimator, prior, x_o, device=device
    )
    theta_transform = mcmc_transform(
        prior, device=device, enable_transform=enable_transform
    )

    return potential_fn, theta_transform


class LikelihoodBasedPotential(BasePotential):
    allow_iid_x = True  # type: ignore

    def __init__(
        self,
        likelihood_estimator: nn.Module,
        prior: Distribution,
        x_o: Optional[Tensor],
        device: str = "cpu",
    ):
        r"""Returns the potential function for likelihood-based methods.

        Args:
            likelihood_estimator: The neural network modelling the likelihood.
            prior: The prior distribution.
            x_o: The observed data at which to evaluate the likelihood.
            device: The device to which parameters and data are moved before evaluating
                the `likelihood_nn`.

        Returns:
            The potential function $p(x_o|\theta)p(\theta)$.
        """

        super().__init__(prior, x_o, device)
        self.likelihood_estimator = likelihood_estimator
        self.likelihood_estimator.eval()

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        r"""Returns the potential $\log(p(x_o|\theta)p(\theta))$.

        Args:
            theta: The parameter set at which to evaluate the potential function.
            track_gradients: Whether to track the gradients.

        Returns:
            The potential $\log(p(x_o|\theta)p(\theta))$.
        """

        # Calculate likelihood over trials and in one batch.
        log_likelihood_trial_sum = _log_likelihoods_over_trials(
            x=self.x_o,
            theta=theta.to(self.device),
            net=self.likelihood_estimator,
            track_gradients=track_gradients,
        )

        return log_likelihood_trial_sum + self.prior.log_prob(theta)


def _log_likelihoods_over_trials(
    x: Tensor, theta: Tensor, net: Any, track_gradients: bool = False
) -> Tensor:
    r"""Return log likelihoods summed over iid trials of `x`.

    Note: `x` can be a batch with batch size larger 1. Batches in `x` are assumed
    to be iid trials, i.e., data generated based on the same paramters /
    experimental conditions.

    Repeats `x` and $\theta$ to cover all their combinations of batch entries.

    Args:
        x: batch of iid data.
        theta: batch of parameters
        net: neural net with .log_prob()
        track_gradients: Whether to track gradients.

    Returns:
        log_likelihood_trial_sum: log likelihood for each parameter, summed over all
            batch entries (iid trials) in `x`.
    """

    # Repeat `x` in case of evaluation on multiple `theta`. This is needed below in
    # when calling nflows in order to have matching shapes of theta and context x
    # at neural network evaluation time.
    theta_repeated, x_repeated = match_theta_and_x_batch_shapes(
        theta=atleast_2d(theta), x=atleast_2d(x)
    )
    assert (
        x_repeated.shape[0] == theta_repeated.shape[0]
    ), "x and theta must match in batch shape."
    assert (
        next(net.parameters()).device == x.device and x.device == theta.device
    ), f"""device mismatch: net, x, theta: {next(net.parameters()).device}, {x.device},
        {theta.device}."""

    # Calculate likelihood in one batch.
    with torch.set_grad_enabled(track_gradients):
        log_likelihood_trial_batch = net.log_prob(x_repeated, theta_repeated)
        # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
        log_likelihood_trial_sum = log_likelihood_trial_batch.reshape(
            x.shape[0], -1
        ).sum(0)

    return log_likelihood_trial_sum


def mixed_likelihood_estimator_based_potential(
    likelihood_estimator: MixedDensityEstimator,
    prior: Distribution,
    x_o: Optional[Tensor],
) -> Tuple[Callable, TorchTransform]:
    r"""Returns $\log(p(x_o|\theta)p(\theta))$ for mixed likelihood-based methods.

    It also returns a transformation that can be used to transform the potential into
    unconstrained space.

    Args:
        likelihood_estimator: The neural network modelling the likelihood.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the likelihood.

    Returns:
        The potential function $p(x_o|\theta)p(\theta)$ and a transformation that maps
        to unconstrained space.
    """

    device = str(next(likelihood_estimator.discrete_net.parameters()).device)

    potential_fn = MixedLikelihoodBasedPotential(
        likelihood_estimator, prior, x_o, device=device
    )
    theta_transform = mcmc_transform(prior, device=device)

    return potential_fn, theta_transform


class MixedLikelihoodBasedPotential(LikelihoodBasedPotential):
    def __init__(
        self,
        likelihood_estimator: MixedDensityEstimator,
        prior: Distribution,
        x_o: Optional[Tensor],
        device: str = "cpu",
    ):
        super().__init__(likelihood_estimator, prior, x_o, device)

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        #print("yo", theta.shape, self.x_o.shape)
        # Calculate likelihood in one batch.
        with torch.set_grad_enabled(track_gradients):
            # Call the specific log prob method of the mixed likelihood estimator as
            # this optimizes the evaluation of the discrete data part.
            # TODO: how to fix pyright issues?
            log_likelihood_trial_batch = self.likelihood_estimator.log_prob_iid(
                x=self.x_o,
                theta=theta.to(self.device),
            )  # type: ignore
            # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
            log_likelihood_trial_sum = log_likelihood_trial_batch.reshape(
                self.x_o.shape[0], -1
            ).sum(0)

        return log_likelihood_trial_sum + self.prior.log_prob(theta)

class MixedLikelihoodBasedPotential(LikelihoodBasedPotential):
    def __init__(
        self,
        likelihood_estimator: MixedDensityEstimator,
        prior: Distribution,
        x_o: Optional[Tensor],
        device: str = "cpu",
        # print_: Optional[bool] = False,
    ):
        super().__init__(likelihood_estimator, prior, x_o, device)

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        #print("theta and x_o shapes", theta.shape, self.x_o.shape)
        # Calculate likelihood in one batch.
        with torch.set_grad_enabled(track_gradients):
            # Call the specific log prob method of the mixed likelihood estimator as
            # this optimizes the evaluation of the discrete data part.
            # TODO: how to fix pyright issues?
            log_likelihood_trial_batch = self.likelihood_estimator.log_prob_iid(
                x=self.x_o,
                theta=theta.to(self.device),
            )  # type: ignore
            # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
            log_likelihood_trial_sum = log_likelihood_trial_batch.reshape(
                self.x_o.shape[0], -1
            ).sum(0)

        if print_:
            print("prior theta shape", theta.shape)
            print("prior theta", theta)
            print("prior lp", self.prior.log_prob(theta))

            print("ll theta shape", theta.shape)
            print("ll theta", theta)
            print("ll sum", log_likelihood_trial_sum)
    
        return log_likelihood_trial_sum + self.prior.log_prob(theta)

class ConditionedMixedLikelihoodBasedPotential(LikelihoodBasedPotential):
    def __init__(
        self,
        likelihood_estimator: MixedDensityEstimator,
        prior: Distribution,
        x_o: Optional[Tensor],
        experimental_conditions: Optional[Tensor],
        device: str = "cpu",
        #print_: Optional[bool] = False,
    ):
        self.experimental_conditions = experimental_conditions
        super().__init__(likelihood_estimator, prior, x_o, device)

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        num_chains = theta.shape[0]
        num_trials = self.x_o.shape[0]

        num_exp_conds = self.experimental_conditions.shape[1]
        # abls = self.experimental_conditions["abl"]
        # ilds = self.experimental_conditions["ild"]

        aux_theta = self.experimental_conditions  # tensor with dim num_trials x num_exp_conds 
        aux_theta = aux_theta.repeat(num_chains,1)  # dim num_trials.num_chains x exp_conds

        #print("aux theta shape", aux_theta.shape)

        conditional_theta = torch.repeat_interleave(theta, num_trials, dim=0)
        #aux_theta = torch.vstack((abls,ilds)).T.repeat(num_chains,1)

        conditional_theta[:,-num_exp_conds:] = aux_theta

        # Calculate likelihood in one batch.
        with torch.set_grad_enabled(track_gradients):
            # Call the specific log prob method of the mixed likelihood estimator as
            # this optimizes the evaluation of the discrete data part.
            # TODO: how to fix pyright issues?
            # log_likelihood_trial_batch = self.likelihood_estimator.log_prob_iid(
            #     x=self.x_o,
            #     theta=theta.to(self.device),
            # )  # type: ignore

            log_likelihood_trial_batch = self.likelihood_estimator.log_prob(
                x=self.x_o.repeat(num_chains,1),
                context=conditional_theta.to(self.device),
            )  # type: ignore

            # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
            log_likelihood_trial_sum = log_likelihood_trial_batch.reshape(
                num_trials, -1
            ).sum(0)

        if print_:
            print("prior theta shape", theta.shape)
            print("prior theta", theta)
            print("prior lp", self.prior.log_prob(theta))

            print("ll cond theta shape", conditional_theta.shape)
            print("ll cond theta", conditional_theta)
            print("ll sum", log_likelihood_trial_sum)

        return log_likelihood_trial_sum + self.prior.log_prob(theta)
    
print_ = False

class ConditionedMixedLikelihoodBasedPotential4(LikelihoodBasedPotential):
    def __init__(
        self,
        likelihood_estimator: MixedDensityEstimator,
        prior: Distribution,
        x_o: Optional[Tensor],
        experimental_conditions: Optional[Tensor],
        device: str = "cpu",
    ):
        self.experimental_conditions = experimental_conditions
        super().__init__(likelihood_estimator, prior, x_o, device)

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        num_chains = theta.shape[0]
        num_trials = self.x_o.shape[0]

        num_exp_conds = self.experimental_conditions.shape[1]
        # abls = self.experimental_conditions["abl"]
        # ilds = self.experimental_conditions["ild"]

        aux_theta = self.experimental_conditions  # tensor with dim num_trials x num_exp_conds 
        aux_theta = aux_theta.repeat(num_chains,1)  # dim num_trials.num_chains x exp_conds

        #print("aux theta shape", aux_theta.shape)

        conditional_theta = torch.repeat_interleave(theta, num_trials, dim=0)
        #aux_theta = torch.vstack((abls,ilds)).T.repeat(num_chains,1)

        conditional_theta[:,-num_exp_conds:] = aux_theta

        # Calculate likelihood in one batch.
        with torch.set_grad_enabled(track_gradients):
            # Call the specific log prob method of the mixed likelihood estimator as
            # this optimizes the evaluation of the discrete data part.
            # TODO: how to fix pyright issues?
            # log_likelihood_trial_batch = self.likelihood_estimator.log_prob_iid(
            #     x=self.x_o,
            #     theta=theta.to(self.device),
            # )  # type: ignore

            log_likelihood_trial_batch = self.likelihood_estimator.log_prob(
                x=self.x_o.repeat(num_chains,1),
                context=conditional_theta.to(self.device),
            )  # type: ignore

            # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
            log_likelihood_trial_sum = log_likelihood_trial_batch.reshape(
                num_trials, -1
            ).sum(0)

        if print_:
            print("prior cond theta shape", conditional_theta.shape)
            print("prior cond theta", conditional_theta)
            print("prior lp", self.prior.log_prob(conditional_theta).sum())

            print("ll cond theta shape", conditional_theta.shape)
            print("ll cond theta", conditional_theta)
            print("ll sum", log_likelihood_trial_sum)

        return log_likelihood_trial_sum + self.prior.log_prob(conditional_theta).sum()


class ConditionedMixedLikelihoodBasedPotential3(LikelihoodBasedPotential):
    def __init__(
        self,
        likelihood_estimator: MixedDensityEstimator,
        prior: Distribution,
        x_o: Optional[Tensor],
        experimental_conditions: Optional[Tensor],
        device: str = "cpu",
        #print_: Optional[bool] = False,
    ):
        self.experimental_conditions = experimental_conditions
        super().__init__(likelihood_estimator, prior, x_o, device)

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        num_chains = theta.shape[0]
        num_trials = self.x_o.shape[0]

        num_exp_conds = self.experimental_conditions.shape[1]
        num_params = theta.shape[1]
        num_model_params = num_params - num_exp_conds
        # abls = self.experimental_conditions["abl"]
        # ilds = self.experimental_conditions["ild"]

        aux_theta = self.experimental_conditions  # tensor with dim num_trials x num_exp_conds 
        aux_theta = aux_theta.repeat(num_chains,1)  # dim num_trials.num_chains x exp_conds

        #print("aux theta shape", aux_theta.shape)

        conditional_theta = torch.repeat_interleave(theta, num_trials, dim=0)
        #aux_theta = torch.vstack((abls,ilds)).T.repeat(num_chains,1)

        conditional_theta[:,-num_exp_conds:] = aux_theta

        # Calculate likelihood in one batch.
        with torch.set_grad_enabled(track_gradients):
            # Call the specific log prob method of the mixed likelihood estimator as
            # this optimizes the evaluation of the discrete data part.
            # TODO: how to fix pyright issues?
            # log_likelihood_trial_batch = self.likelihood_estimator.log_prob_iid(
            #     x=self.x_o,
            #     theta=theta.to(self.device),
            # )  # type: ignore

            log_likelihood_trial_batch = self.likelihood_estimator.log_prob(
                x=self.x_o.repeat(num_chains,1),
                context=conditional_theta.to(self.device),
            )  # type: ignore

            # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
            log_likelihood_trial_sum = log_likelihood_trial_batch.reshape(
                num_trials, -1
            ).sum(0)

        if print_:
            print("prior trimmed theta shape", theta[:,:num_model_params].shape)
            print("prior trimmed theta", theta[:,:num_model_params])
            print("prior lp",  self.prior.log_prob(theta[:,:num_model_params]))

            print("ll cond theta shape", conditional_theta.shape)
            print("ll cond theta", conditional_theta)
            print("ll sum", log_likelihood_trial_sum)

        return log_likelihood_trial_sum + self.prior.log_prob(theta[:,:num_model_params])
    

class ConditionedMixedLikelihood(LikelihoodBasedPotential):
    def __init__(
        self,
        likelihood_estimator: MixedDensityEstimator,
        prior: Distribution,
        x_o: Optional[Tensor],
        experimental_conditions: Optional[Tensor],
        device: str = "cpu",
        #print_: Optional[bool] = False,
    ):
        self.experimental_conditions = experimental_conditions
        super().__init__(likelihood_estimator, prior, x_o, device)

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        theta = theta.reshape(1,-1)

        num_trials = self.x_o.shape[0]

        num_exp_conds = self.experimental_conditions.shape[1]
        num_model_params = theta.shape[1]
        num_params = num_model_params + num_exp_conds

        aux_theta = self.experimental_conditions  # tensor with dim num_trials x num_exp_conds 

        _theta = torch.repeat_interleave(theta, num_trials, dim=0)
        conditional_theta = torch.cat((_theta, aux_theta), dim=1)

        #print("cond th shape", conditional_theta.shape)

        # Calculate likelihood in one batch.
        with torch.set_grad_enabled(track_gradients):
            # Call the specific log prob method of the mixed likelihood estimator as
            # this optimizes the evaluation of the discrete data part.
            # TODO: how to fix pyright issues?
            # )  # type: ignore

            log_likelihood_trial_batch = self.likelihood_estimator.log_prob(
                x=self.x_o,
                context=conditional_theta.to(self.device),
            )  # type: ignore

            # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
            log_likelihood_trial_sum = log_likelihood_trial_batch.reshape(
                num_trials, -1
            ).sum(0)

        if print_:
            print("prior trimmed theta shape", theta[:,:num_model_params].shape)
            print("prior trimmed theta", theta[:,:num_model_params])
            print("prior lp",  self.prior.log_prob(theta[:,:num_model_params]))

            print("ll cond theta shape", conditional_theta.shape)
            print("ll cond theta", conditional_theta)
            print("ll sum", log_likelihood_trial_sum)

        return log_likelihood_trial_sum