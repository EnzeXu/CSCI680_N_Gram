public class BoundaryBalanceGroupBlockingHashJoinTransformer extends RuleInsertionTransformer { public BoundaryBalanceGroupBlockingHashJoinTransformer() { super( BalanceAssembly, new BalanceGroupBlockingHashJoinExpression(), BoundaryElementFactory.BOUNDARY_PIPE ); } }