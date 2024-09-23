public class BalanceGroupSplitTriangleExpression extends RuleExpression { public static final FlowElementExpression SHARED_GROUP = new FlowElementExpression( Group.class, Topo.Split ); public static final ElementExpression SHARED_LHS = or( new FlowElementExpression( HashJoin.class ), new FlowElementExpression( Group.class ), new FlowElementExpression( Tap.class ), new FlowElementExpression( Checkpoint.class ) ); public static final ElementExpression SHARED_RHS = or( new FlowElementExpression( HashJoin.class ), new FlowElementExpression( Group.class ), new FlowElementExpression( Tap.class ), new FlowElementExpression( Checkpoint.class ) ); public BalanceGroupSplitTriangleExpression() { super( new SyncPipeExpressionGraph(), new ExpressionGraph() .arc( SHARED_GROUP, ScopeExpression.ANY, SHARED_LHS ) .arc( SHARED_GROUP, ScopeExpression.ANY, SHARED_RHS ) .arc( SHARED_LHS, ScopeExpression.ANY, SHARED_RHS ), new ExpressionGraph() .arcs( new FlowElementExpression( ElementCapture.Primary, Pipe.class, Topo.SplitOnly ) ) ); } }