public class OrdinalScopeExpression extends ScopeExpression { public static final OrdinalScopeExpression ORDINAL_ZERO = new OrdinalScopeExpression( 0 ); public static final OrdinalScopeExpression NOT_ORDINAL_ZERO = new OrdinalScopeExpression( true, 0 ); boolean not = false; int ordinal = 0; public OrdinalScopeExpression( int ordinal ) { this.ordinal = ordinal; } public OrdinalScopeExpression( Applies applies, int ordinal ) { super( applies ); this.ordinal = ordinal; } public OrdinalScopeExpression( boolean not, int ordinal ) { this.not = not; this.ordinal = ordinal; } public OrdinalScopeExpression( Applies applies, boolean not, int ordinal ) { super( applies ); this.not = not; this.ordinal = ordinal; } @Override public boolean applies( PlannerContext plannerContext, ElementGraph elementGraph, Scope scope ) { boolean equals = scope.getOrdinal().equals( ordinal ); if( !not ) return equals; else return !equals; } }