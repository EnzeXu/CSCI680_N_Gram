public class NoCoGroupJoinTapExpressionGraph extends ExpressionGraph { public NoCoGroupJoinTapExpressionGraph() { super( SearchOrder.ReverseDepth, not( OrElementExpression.or( ElementCapture.Primary, new FlowElementExpression( Extent.class ), new FlowElementExpression( CoGroup.class ), new FlowElementExpression( HashJoin.class ), new FlowElementExpression( Tap.class ) ) ) ); } }