public class StreamedOnlySourcesNodeRePartitioner extends ExpressionRulePartitioner { public StreamedOnlySourcesNodeRePartitioner ( ) { super ( PartitionNodes , PartitionSource . PartitionCurrent , new RuleExpression ( new NoGroupJoinMergeBoundaryTapExpressionGraph ( ) , new StreamedOnlySourcesExpressionGraph ( ) ) ) ; } }