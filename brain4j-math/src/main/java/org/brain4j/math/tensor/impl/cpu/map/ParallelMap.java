package org.brain4j.math.tensor.impl.cpu.map;

import org.brain4j.math.lang.DoubleToDoubleFunction;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class ParallelMap extends RecursiveAction {

    private static final int PARALLELISM = Runtime.getRuntime().availableProcessors();
    private static final int PARALLEL_COMPLEXITY_THRESHOLD = 1024;
    private static final int PARALLEL_WORK_THRESHOLD = PARALLELISM;

    private static final int SPLIT_COMPLEXITY_THRESHOLD = 1024;

    private final MapParameters parameters;
    private final int start;
    private final int end;

    public ParallelMap(MapParameters parameters, int start, int end) {
        this.parameters = parameters;
        this.start = start;
        this.end = end;
    }

    @Override
    protected void compute() {
        int work = end - start;
        if (isOverSplitThreshold(work)) {
            int mid = (start + end) >>> 1;
            invokeAll(
                    new ParallelMap(parameters, start, mid),
                    new ParallelMap(parameters, mid, end)
            );
            return;
        }

        DoubleToDoubleFunction function = parameters.function();
        float[] data = parameters.data();

        mapSection(start, end, data, function);
    }

    public static void map(
            DoubleToDoubleFunction function, float[] data, ForkJoinPool pool
    ) {
        int start = 0;
        int end = data.length;

        int work = end - start;
        if (!isOverParallelThreshold(work)) {
            mapSection(start, end, data, function);
            return;
        }

        int parallelism = PARALLELISM;
        int step = work / parallelism;

        MapParameters parameters = new MapParameters(function, data);
        ParallelMap[] actions = new ParallelMap[parallelism];

        int i;
        for (i = 0; i < parallelism - 1; i++) {
            actions[i] = new ParallelMap(parameters, start + (i * step), start + ((i + 1) * step));
        }
        actions[i] = new ParallelMap(parameters, start + (i * step), end);

        ForkJoinTask.invokeAll(actions);
    }

    private static void mapSection(int start, int end, float[] data, DoubleToDoubleFunction function) {
        for (int i = start; i < end; i++) {
            double value = data[i];
            data[i] = (float) function.apply(value);
        }
    }

    private static boolean isOverParallelThreshold(int work) {
        return work > PARALLEL_WORK_THRESHOLD && work > PARALLEL_COMPLEXITY_THRESHOLD;
    }

    private static boolean isOverSplitThreshold(int work) {
        return work > SPLIT_COMPLEXITY_THRESHOLD;
    }

}