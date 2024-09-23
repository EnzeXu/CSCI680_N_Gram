public class BoundedElementMultiGraph extends ElementMultiGraph { public BoundedElementMultiGraph( ElementGraph parentElementGraph, ElementGraph subElementGraph, EnumMultiMap annotations ) { graph = new DirectedMultiGraph( directed( subElementGraph ) ); addParentAnnotations( parentElementGraph ); getAnnotations().addAll( annotations ); bindHeadAndTail( parentElementGraph, subElementGraph ); } @Override protected void addParentAnnotations( ElementGraph parentElementGraph ) { if( !( parentElementGraph instanceof AnnotatedGraph ) || !( (AnnotatedGraph) parentElementGraph ).hasAnnotations() ) return; Set<FlowElement> vertexSet = vertexSet(); EnumMultiMap parentAnnotations = ( (AnnotatedGraph) parentElementGraph ).getAnnotations(); Set<Enum> allKeys = parentAnnotations.getKeys(); for( Enum annotation : allKeys ) { Set<FlowElement> flowElements = (Set<FlowElement>) parentAnnotations.getValues( annotation ); for( FlowElement flowElement : flowElements ) { if( vertexSet.contains( flowElement ) ) getAnnotations().addAll( annotation, flowElements ); } } } protected void bindHeadAndTail( ElementGraph parentElementGraph, ElementGraph subElementGraph ) { Set<FlowElement> sources = ElementGraphs.findSources( subElementGraph, FlowElement.class ); Set<FlowElement> sinks = ElementGraphs.findSinks( subElementGraph, FlowElement.class ); addVertex( head ); addVertex( tail ); Set<FlowElement> parentElements = parentElementGraph.vertexSet(); for( FlowElement source : sources ) { if( !parentElements.contains( source ) ) continue; Set<Scope> scopes = parentElementGraph.incomingEdgesOf( source ); for( Scope scope : scopes ) addEdge( head, source, scope ); } for( FlowElement sink : sinks ) { if( !parentElements.contains( sink ) ) continue; Set<Scope> scopes = parentElementGraph.outgoingEdgesOf( sink ); for( Scope scope : scopes ) addEdge( sink, tail, scope ); } } }