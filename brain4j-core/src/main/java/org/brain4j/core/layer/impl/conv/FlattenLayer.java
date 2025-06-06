package org.brain4j.core.layer.impl.conv;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.math.tensor.Tensor;

public class FlattenLayer extends Layer {

    @Override
    public Tensor forward(int index, StatesCache cache, Tensor input, boolean training) {
        return input.reshape(1, input.elements());
    }
}
