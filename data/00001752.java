public class StreamedOnlySourcesExpressionGraph extends ExpressionGraph { public StreamedOnlySourcesExpressionGraph() { super( SearchOrder.Depth, true ); this.arc( or( ElementCapture.Primary, new FlowElementExpression( Tap.class ), new FlowElementExpression( Group.class ) ), PathScopeExpression.ALL, or( new FlowElementExpression( Tap.class ), new FlowElementExpression( Group.class ) ) ); } }