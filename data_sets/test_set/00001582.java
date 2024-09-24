public class NonBlockedBlockedJoinExpressionGraph extends ExpressionGraph { public NonBlockedBlockedJoinExpressionGraph() { super( SearchOrder.ReverseDepth ); ElementExpression source = OrElementExpression.or( ElementCapture.Primary, new FlowElementExpression( Tap.class ), new FlowElementExpression( Group.class ) ); ElementExpression sink = new FlowElementExpression( ElementCapture.Secondary, HashJoin.class ); this.arc( source, PathScopeExpression.ANY_NON_BLOCKING, sink ); this.arc( source, PathScopeExpression.ANY_BLOCKING, sink ); } }