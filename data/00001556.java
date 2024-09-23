public class ElementSubGraph extends BaseElementGraph implements ElementGraph { private final ElementGraph elementGraph; private final Set<FlowElement> flowElements; private final Set<Scope> scopes; public ElementSubGraph( ElementGraph elementGraph, Collection<FlowElement> flowElements ) { this( elementGraph, flowElements, null ); } public ElementSubGraph( ElementGraph elementGraph, Collection<FlowElement> flowElements, Collection<Scope> scopes ) { this.flowElements = createIdentitySet( flowElements ); this.scopes = scopes == null || scopes.isEmpty() ? null : createIdentitySet( scopes ); this.graph = new DirectedSubGraph( directed( elementGraph ), this.flowElements, this.scopes ); this.elementGraph = elementGraph; } public ElementSubGraph( ElementSubGraph graph ) { this( graph.elementGraph, graph.flowElements, graph.scopes ); } @Override public ElementGraph copyElementGraph() { return new ElementSubGraph( this ); } private class DirectedSubGraph extends DirectedSubgraph<FlowElement, Scope> { public DirectedSubGraph( Graph<FlowElement, Scope> base, Set<FlowElement> vertexSubset, Set<Scope> edgeSubset ) { super( base, vertexSubset, edgeSubset ); } } }