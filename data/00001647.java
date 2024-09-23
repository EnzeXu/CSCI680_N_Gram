public class MapReduceHadoopRuleRegistry extends RuleRegistry { public MapReduceHadoopRuleRegistry ( ) { addRule ( new LoneGroupAssert ( ) ) ; addRule ( new MissingGroupAssert ( ) ) ; addRule ( new BufferAfterEveryAssert ( ) ) ; addRule ( new EveryAfterBufferAssert ( ) ) ; addRule ( new SplitBeforeEveryAssert ( ) ) ; addRule ( new TapBalanceGroupSplitTransformer ( ) ) ; addRule ( new TapBalanceGroupSplitJoinTransformer ( ) ) ; addRule ( new TapBalanceGroupSplitTriangleTransformer ( ) ) ; addRule ( new TapBalanceGroupSplitMergeGroupTransformer ( ) ) ; addRule ( new TapBalanceGroupSplitMergeTransformer ( ) ) ; addRule ( new TapBalanceGroupMergeGroupTransformer ( ) ) ; addRule ( new TapBalanceGroupGroupTransformer ( ) ) ; addRule ( new TapBalanceCheckpointTransformer ( ) ) ; addRule ( new TapBalanceHashJoinSameSourceTransformer ( ) ) ; addRule ( new TapBalanceHashJoinBlockingHashJoinTransformer ( ) ) ; addRule ( new TapBalanceGroupBlockingHashJoinTransformer ( ) ) ; addRule ( new TapBalanceGroupNonBlockingHashJoinTransformer ( ) ) ; addRule ( new TapBalanceSameSourceStreamedAccumulatedTransformer ( ) ) ; addRule ( new TapBalanceNonSafeSplitTransformer ( ) ) ; addRule ( new TapBalanceNonSafePipeSplitTransformer ( ) ) ; addRule ( new RemoveNoOpPipeTransformer ( ) ) ; addRule ( new ApplyAssertionLevelTransformer ( ) ) ; addRule ( new ApplyDebugLevelTransformer ( ) ) ; addRule ( new ReplaceAccumulateTapWithDistCacheTransformer ( ) ) ; addRule ( new ConsecutiveTapsStepPartitioner ( ) ) ; addRule ( new TapGroupTapStepPartitioner ( ) ) ; addRule ( new ConsecutiveTapsNodePartitioner ( ) ) ; addRule ( new MultiTapGroupNodePartitioner ( ) ) ; addRule ( new GroupTapNodePartitioner ( ) ) ; addRule ( new StreamedAccumulatedTapsHashJoinPipelinePartitioner ( ) ) ; addRule ( new StreamedAccumulatedTapsPipelinePartitioner ( ) ) ; addRule ( new StreamedSelfJoinSourcesPipelinePartitioner ( ) ) ; addRule ( new StreamedOnlySourcesPipelinePartitioner ( ) ) ; addRule ( new RemoveMalformedHashJoinPipelineTransformer ( ) ) ; } }