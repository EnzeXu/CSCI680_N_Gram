public class TestNoTapExpressionGraph extends ExpressionGraph { public TestNoTapExpressionGraph() { super( SearchOrder.ReverseDepth, not( or( ElementCapture.Primary, new FlowElementExpression( Extent.class ), new FlowElementExpression( Tap.class ) ) ) ); } }