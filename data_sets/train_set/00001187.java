public class BottomUpJoinedBoundariesNodePartitioner extends ExpressionRulePartitioner { public BottomUpJoinedBoundariesNodePartitioner() { super( PartitionNodes, new RuleExpression( new NoGroupJoinMergeBoundaryTapExpressionGraph(), new BottomUpConsecutiveBoundariesExpressionGraph() ), new ElementAnnotation( ElementCapture.Primary, IORole.sink ) ); } }