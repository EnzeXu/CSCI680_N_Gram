public class MultiTapGroupExpressionGraph extends ExpressionGraph { public MultiTapGroupExpressionGraph() { super( SearchOrder.ReverseTopological ); this .arc( new FlowElementExpression( Tap.class ), ScopeExpression.ALL, new FlowElementExpression( ElementCapture.Primary, Group.class ) ); } }