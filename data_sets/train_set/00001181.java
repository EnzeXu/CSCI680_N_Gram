public class TopDownBoundariesNodePartitioner extends ExpressionRulePartitioner { public TopDownBoundariesNodePartitioner() { super( PartitionNodes, new RuleExpression( new NoGroupJoinMergeBoundaryTapExpressionGraph(), new TopDownConsecutiveBoundariesExpressionGraph() ), new ElementAnnotation( ElementCapture.Include, IORole.sink ) ); } }