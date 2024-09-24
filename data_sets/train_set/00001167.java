public class BottomUpNoSplitConsecutiveBoundariesExpressionGraph extends ExpressionGraph { public BottomUpNoSplitConsecutiveBoundariesExpressionGraph() { super( SearchOrder.ReverseTopological ); this.arc( or( new FlowElementExpression( Boundary.class, TypeExpression.Topo.LinearOut ), new FlowElementExpression( Tap.class, TypeExpression.Topo.LinearOut ), new FlowElementExpression( Group.class, TypeExpression.Topo.LinearOut ), new FlowElementExpression( Merge.class, TypeExpression.Topo.LinearOut ) ), PathScopeExpression.ANY, new BoundariesElementExpression( ElementCapture.Primary ) ); } }