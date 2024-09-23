public class EquivalentTapsScopeExpression extends ScopeExpression { @Override public boolean applies( PlannerContext plannerContext, ElementGraph elementGraph, Scope scope ) { FlowElement edgeSource = elementGraph.getEdgeSource( scope ); FlowElement edgeTarget = elementGraph.getEdgeTarget( scope ); if( !( edgeSource instanceof Hfs ) || !( edgeTarget instanceof Hfs ) ) throw new IllegalStateException( "non Hfs Taps matched" ); Hfs predecessor = (Hfs) edgeSource; Hfs successor = (Hfs) edgeTarget; if( !successor.getScheme().isSymmetrical() ) return false; HadoopPlanner flowPlanner = (HadoopPlanner) plannerContext.getFlowPlanner(); URI tempURIScheme = flowPlanner.getDefaultURIScheme( predecessor ); URI successorURIScheme = flowPlanner.getURIScheme( successor ); if( !tempURIScheme.equals( successorURIScheme ) ) return false; if( !predecessor.getSourceFields().equals( successor.getSourceFields() ) ) return true; return true; } }