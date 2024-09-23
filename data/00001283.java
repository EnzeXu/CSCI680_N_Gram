public class TestConsecutiveTapsExpressionGraph extends ExpressionGraph { ElementExpression shared = or( ElementCapture.Secondary, new FlowElementExpression( Tap.class ), new FlowElementExpression( Group.class ) ); public TestConsecutiveTapsExpressionGraph() { super( SearchOrder.Depth ); this.arc( new FlowElementExpression( ElementCapture.Primary, Tap.class ), PathScopeExpression.ALL_NON_BLOCKING, shared ); this.arc( new FlowElementExpression( Tap.class ), PathScopeExpression.ANY_BLOCKING, shared ); } }