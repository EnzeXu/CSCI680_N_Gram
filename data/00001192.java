public class BoundaryBalanceHashJoinToHashJoinTransformer extends RuleInsertionTransformer { public BoundaryBalanceHashJoinToHashJoinTransformer() { super( BalanceAssembly, new BalanceHashJoinToHashJoinExpression(), BoundaryElementFactory.BOUNDARY_PIPE ); } }