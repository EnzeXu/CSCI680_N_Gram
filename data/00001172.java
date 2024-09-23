public class BottomUpConsecutiveBoundariesExpressionGraph extends ExpressionGraph { public BottomUpConsecutiveBoundariesExpressionGraph() { super( SearchOrder.ReverseTopological ); ElementExpression head = or( new FlowElementExpression( Boundary.class ), new FlowElementExpression( Tap.class ), new FlowElementExpression( Group.class, TypeExpression.Topo.LinearOut ), new FlowElementExpression( Merge.class, TypeExpression.Topo.LinearOut ) ); FlowElementExpression shared = new FlowElementExpression( ElementCapture.Secondary, HashJoin.class ); ElementExpression tail = or( ElementCapture.Primary, and( new BoundariesElementExpression( TypeExpression.Topo.LinearIn ), not( new AnnotationExpression( IORole.sink ) ) ), new BoundariesElementExpression( TypeExpression.Topo.Splice ) ); this.arc( head, PathScopeExpression.ANY, shared ); this.arc( shared, PathScopeExpression.ANY, tail ); } }