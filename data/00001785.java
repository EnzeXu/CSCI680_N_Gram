public class BalanceNonSafePipeSplitExpression extends RuleExpression { public BalanceNonSafePipeSplitExpression() { super( new NonSafeAndSplitAndSyncPipeExpressionGraph(), new ExpressionGraph() .arcs( new FlowElementExpression( Tap.class ), new NonSafeOperationExpression(), new FlowElementExpression( Pipe.class, TypeExpression.Topo.SplitOnly ) ), new ExpressionGraph() .arcs( new FlowElementExpression( ElementCapture.Primary, Pipe.class, TypeExpression.Topo.Tail ) ) ); } }