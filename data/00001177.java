public class BoundarySelJoinCoGroupExpressionGraph extends ExpressionGraph { public BoundarySelJoinCoGroupExpressionGraph() { super( SearchOrder.ReverseTopological ); this .arc( new BoundariesElementExpression( ElementCapture.Primary, TypeExpression.Topo.SplitOnly ), ScopeExpression.ALL, new FlowElementExpression( CoGroup.class, TypeExpression.Topo.SpliceOnly ) ); } }