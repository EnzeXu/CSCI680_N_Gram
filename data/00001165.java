public class TopDownSplitBoundariesExpressionGraph extends ExpressionGraph { public TopDownSplitBoundariesExpressionGraph() { super( SearchOrder.Topological ); this.arc( new BoundariesElementExpression( ElementCapture.Primary, TypeExpression.Topo.Split ), PathScopeExpression.ANY, new BoundariesElementExpression() ); } }