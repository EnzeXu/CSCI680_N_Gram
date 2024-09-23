public class NonBlockedBlockedJoinJoinExpressionGraph extends ExpressionGraph { public NonBlockedBlockedJoinJoinExpressionGraph() { super( SearchOrder.ReverseDepth ); ElementExpression source = OrElementExpression.or( ElementCapture.Primary, new FlowElementExpression( Tap.class ), new FlowElementExpression( Group.class ) ); ElementExpression blocking = or( new FlowElementExpression( HashJoin.class ), new FlowElementExpression( Group.class ) ); ElementExpression sink = new FlowElementExpression( ElementCapture.Secondary, HashJoin.class ); this.arc( source, PathScopeExpression.ANY, blocking ); this.arc( blocking, PathScopeExpression.ANY, sink ); this.arc( source, PathScopeExpression.ANY, sink ); } }