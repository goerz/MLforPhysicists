import numpy as np
import attr

import matplotlib.pylab as plt
from visualize_network import render_network, render_21_output


def column_vec(a):
    """Convert `a` into a numpy column vector.

    Note numpy is pretty smart when adding column vectors to other arrays of
    various dimenions: Adding a column vector to a matrix adds it to every
    column of the matrix. Adding a 1-entry column vector (a scalar) to a vector
    or matrix adds the scalar value to all entries. Thus, we get exactly the
    behavior we need for the bias vector in a neural net.
    """
    try:
        n = len(a)
    except TypeError:
        # unsized object (scalar)
        n = 1
    return np.array(a).reshape((n, 1))


@attr.s
class Layer:
    W = attr.ib(converter=np.array)
    b = attr.ib(converter=column_vec)
    f = attr.ib(validator=attr.validators.is_callable())

    @property
    def n_inputs(self):
        """The number of inputs to the layer."""
        return self.W.shape[1]

    @property
    def n_nodes(self):
        """The number of nodes (outputs) in the layer."""
        return self.W.shape[0]

    @property
    def n_weights(self):
        """The number of weight parameters in the layer."""
        return self.W.size

    def evaluate(self, y0):
        """Evaluate the layer for the given input values."""
        return self.f(self.W @ y0 + self.b)

    def check(self, _raise=True):
        """Check attributes for internal consistency.

        Raises:
            ValueError: if bias vector inconsistent with weights and `_raise`
            is True.
        """
        if self.b.shape != (1, 1) and self.b.shape != (self.n_nodes, 1):
            if _raise:
                raise ValueError(
                    "bias vector shape %s is incompatible with "
                    "weight matrix shape %s." % (self.b.shape, self.W.shape)
                )
            else:
                return False
        return True


class NeuralNet:
    """Neural net consisting of the given layers.

    The layers are ordered bottom to top, and exclude the input layer.
    """

    def __init__(self, *layers):
        self.layers = layers
        self.check()

    @property
    def weights(self):
        """List of weight-matrices for all layers."""
        return [l.W for l in self.layers]

    @property
    def biases(self):
        """List of biases for all layers."""
        return [l.b for l in self.layers]

    @property
    def n_inputs(self):
        return self.layers[0].n_inputs

    @property
    def n_outputs(self):
        return self.layers[-1].n_nodes

    @property
    def n_nodes(self):
        """The total number of nodes in the network."""
        return sum((l.n_nodes for l in self.layers), self.n_inputs)

    @property
    def n_weights(self):
        """The total number of wight parameters in the network."""
        return sum(l.n_weights for l in self.layers)

    @property
    def n_layers(self):
        """The number of layers in the network (including input layer)."""
        return len(self.layers) + 1

    def evaluate(self, y0):
        """Evaluate the network for the given input values.

        To evaluate the network for a list of vectors in parallel, `y0` may be
        a matrix of column vectors.
        """
        for l in self.layers:
            y0 = l.evaluate(y0)
        return y0

    def remove_hidden_layer(self, index=-1):
        """Return a new network with the given hidden layer index removed.

        The previous and next hidden layers must have a compatible number of
        inputs/outputs for this to work.

        Args:
            index: the 0-based index for the hidden layer to remove. A value of
            -1 (default) removes the hidden layer prior to the output layer.

        Raises:
            ValueError: If removing the hidden layer produces an invalid
            network, due to a mismatch in the input/outputs of the remaining
            hidden layers.
        """
        *hidden_layers, output_layer = self.layers
        return NeuralNet(
            *hidden_layers[0:index], *hidden_layers[index:][1:], output_layer,
        )

    def with_hidden_activation_function(self, f):
        """Set the activation function for all hidden layers."""
        *hidden_layers, output_layer = self.layers
        hidden_layers = [Layer(l.W, l.b, f) for l in hidden_layers]
        return NeuralNet(*hidden_layers, output_layer)

    def with_output_activation_function(self, f):
        """Set the activation function for the output layer."""
        output_layer = self.layers[-1]
        W = output_layer.W
        b = output_layer.b
        return NeuralNet(*self.layers[:-1], Layer(W, b, f=f))

    def scale(self, factor, scale_weights=True, scale_biases=True):
        """A network with all weights and biases scaled by a constant `factor`.
        """
        W_factor = 1
        if scale_weights:
            W_factor = factor
        b_factor = 1
        if scale_biases:
            b_factor = factor
        return NeuralNet(
            *[Layer(W_factor * l.W, b_factor * l.b, l.f) for l in self.layers]
        )

    def shift_bias(self, shift_by, shift_hidden=True, shift_output=True):
        """A network with shifted bias values.

        If `shift_hidden` is True, shift the bias value of all hidden layer
        nodes. If `shift_output` is True, shift the bias value of nodes in the
        output layer.
        """
        *hidden_layers, output_layer = self.layers
        if shift_hidden:
            hidden_layers = [
                Layer(l.W, l.b + shift_by, l.f) for l in hidden_layers
            ]
        if shift_output:
            output_layer = Layer(
                output_layer.W, output_layer.b + shift_by, output_layer.f
            )
        return NeuralNet(*hidden_layers, output_layer)

    def check(self, n_inputs=None, n_outputs=None, _raise=True):
        """Check internal consistency of all layers.

        Raises:
            ValueError: if any layers have inconsistencies, if `_raise` is
            True.
        """
        if n_inputs is not None:
            if self.n_inputs != n_inputs:
                if _raise:
                    raise ValueError(
                        "The network has %d inputs instead of the expected "
                        "%d inputs" % (self.n_inputs, n_inputs)
                    )
                else:
                    return False
        prev_layer_outputs = self.n_inputs
        for (i, l) in enumerate(self.layers, start=1):
            try:
                l.check()
            except ValueError as exc_info:
                if _raise:
                    raise ValueError("Layer %d: %s" % exc_info)
                else:
                    return False
            if l.n_inputs != prev_layer_outputs:
                if _raise:
                    raise ValueError(
                        "Layer %d: number of inputs %d incompatible with "
                        "number of output %d from previous layer"
                        % (l.n_inputs, prev_layer_outputs)
                    )
                else:
                    return False
            prev_layer_outputs = l.n_nodes
        if n_outputs is not None:
            if self.n_outputs != n_outputs:
                if _raise:
                    raise ValueError(
                        "The network has %d outputs instead of the expected "
                        "%d outputs" % (self.n_outputs, n_outputs)
                    )
                else:
                    return False
        return True

    def visualize(
        self, node_size=400, linewidth=5.0, figsize=(4, 4), show=True, ax=None
    ):
        """Render a diagram of the network.

        node_size: Size of marker for network nodes, in points**2
        linewidth: the width of the connection lines in the network, in points
        figsize: The size of the figure to create, in inches
        show: If true, display the resulting plot. Otherwise, return the
            resulting Figure.
        ax: Optional pre-existing Axes instance.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        bias_vectors = [
            # adding a zero vector converts scalars into vectors
            l.b.flatten() + np.zeros(l.n_nodes)
            for l in self.layers
        ]
        render_network(
            ax,
            self.weights,
            bias_vectors,
            size=node_size,
            linewidth=linewidth,
        )
        if fig is not None:
            if show:
                fig.show()
            else:
                return fig

    def show_output(
        self,
        *ranges,
        samples=100,
        figsize=(4, 4),
        cmap=None,
        show=True,
        fig=None,
        ax=None,
        fig_title=None,
    ):
        """Plot the output of the network.

        ranges: for each input node, the range of values to visualize
        samples: the number of sampling points to cover the `ranges
        figsize: The size of the figure to create, in inches
        cmap: The colormap to use
        show: If True, render the plot. Otherwise, return the resulting Figure
            instance.
        fig: Optional pre-existing Figure instance. Must be given together with
            `ax`.
        ax: Optional pre-existing Axes instance.
        """
        if self.n_inputs != 2 and self.n_outputs != 1:
            raise NotImplementedError(
                "Only implemented for 2 inputs and 1 output"
            )
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        try:
            y0range = ranges[0]
            y1range = ranges[1]
        except IndexError:
            y0range = [-1, 1]
            y1range = [-1, 1]
        y0, y1 = np.meshgrid(
            np.linspace(y0range[0], y0range[1], samples),
            np.linspace(y1range[0], y1range[1], samples),
        )
        y_in = np.zeros(shape=(2, samples * samples))
        y_in[0, :] = y0.flatten()
        y_in[1, :] = y1.flatten()
        y_out = self.evaluate(y_in)

        render_21_output(
            ax,
            fig,
            y_out,
            M=samples,
            y0range=y0range,
            y1range=y1range,
            cmap=cmap,
        )
        if fig_title is not None:
            fig.suptitle(fig_title)
        if show:
            fig.show()
        else:
            return fig


def visualize_2to1_network(
    network, *ranges, figsize=(8, 4), node_size=400, samples=100, linewidth=5
):
    """Combination of `visualize` and `show_output` for a network with two
    inputs and one output.

    Arguments are forwarded to the respective methods.
    """
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=figsize)
    network.visualize(ax=axs[0], node_size=node_size, linewidth=linewidth)
    network.show_output(*ranges, ax=axs[1], fig=fig, samples=100)
