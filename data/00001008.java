public class DualStreamedAccumulatedMergeNodeAssert extends RuleAssert { public DualStreamedAccumulatedMergeNodeAssert ( ) { super ( PostNodes , new DualStreamedAccumulatedExpression ( ) , "may not merge accumulated and streamed in same pipeline : { Secondary } " ) ; } }