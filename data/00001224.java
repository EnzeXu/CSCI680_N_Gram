public class DOTExporter < V , E > { private VertexNameProvider < V > vertexIDProvider ; private VertexNameProvider < V > vertexLabelProvider ; private EdgeNameProvider < E > edgeLabelProvider ; private ComponentAttributeProvider < V > vertexAttributeProvider ; private ComponentAttributeProvider < E > edgeAttributeProvider ; public DOTExporter ( ) { this ( new IntegerNameProvider < V > ( ) , null , null ) ; } public DOTExporter ( VertexNameProvider < V > vertexIDProvider , VertexNameProvider < V > vertexLabelProvider , EdgeNameProvider < E > edgeLabelProvider ) { this ( vertexIDProvider , vertexLabelProvider , edgeLabelProvider , null , null ) ; } public DOTExporter ( VertexNameProvider < V > vertexIDProvider , VertexNameProvider < V > vertexLabelProvider , EdgeNameProvider < E > edgeLabelProvider , ComponentAttributeProvider < V > vertexAttributeProvider , ComponentAttributeProvider < E > edgeAttributeProvider ) { this . vertexIDProvider = vertexIDProvider ; this . vertexLabelProvider = vertexLabelProvider ; this . edgeLabelProvider = edgeLabelProvider ; this . vertexAttributeProvider = vertexAttributeProvider ; this . edgeAttributeProvider = edgeAttributeProvider ; } public void export ( Writer writer , Graph < V , E > g ) { PrintWriter out = new PrintWriter ( writer ) ; String indent = " " ; String connector ; if ( g instanceof DirectedGraph < ? , ? > ) { out . println ( "digraph G { " ) ; connector = " - > " ; } else { out . println ( "graph G { " ) ; connector = " -- " ; } for ( V v : g . vertexSet ( ) ) { out . print ( indent + getVertexID ( v ) ) ; String labelName = null ; if ( vertexLabelProvider != null ) { labelName = vertexLabelProvider . getVertexName ( v ) ; } Map < String , String > attributes = null ; if ( vertexAttributeProvider != null ) { attributes = vertexAttributeProvider . getComponentAttributes ( v ) ; } renderAttributes ( out , labelName , attributes ) ; out . println ( " ; " ) ; } for ( E e : g . edgeSet ( ) ) { String source = getVertexID ( g . getEdgeSource ( e ) ) ; String target = getVertexID ( g . getEdgeTarget ( e ) ) ; out . print ( indent + source + connector + target ) ; String labelName = null ; if ( edgeLabelProvider != null ) { labelName = edgeLabelProvider . getEdgeName ( e ) ; } Map < String , String > attributes = null ; if ( edgeAttributeProvider != null ) { attributes = edgeAttributeProvider . getComponentAttributes ( e ) ; } renderAttributes ( out , labelName , attributes ) ; out . println ( " ; " ) ; } out . println ( " } " ) ; out . flush ( ) ; } private void renderAttributes ( PrintWriter out , String labelName , Map < String , String > attributes ) { if ( ( labelName == null ) && ( attributes == null ) ) { return ; } out . print ( " [ " ) ; if ( ( labelName == null ) && ( attributes != null ) ) { labelName = attributes . get ( "label" ) ; } if ( labelName != null ) { out . print ( "label=\"" + labelName + "\" " ) ; } if ( attributes != null ) { for ( Map . Entry < String , String > entry : attributes . entrySet ( ) ) { String name = entry . getKey ( ) ; if ( name . equals ( "label" ) ) { continue ; } out . print ( name + "=\"" + entry . getValue ( ) + "\" " ) ; } } out . print ( " ] " ) ; } private String getVertexID ( V v ) { String idCandidate = vertexIDProvider . getVertexName ( v ) ; boolean isAlphaDig = idCandidate . matches ( " [ a-zA-Z ] + ( [ \\w_ ] * ) ?" ) ; boolean isDoubleQuoted = idCandidate . matches ( "\" . *\"" ) ; boolean isDotNumber = idCandidate . matches ( " [ - ] ? ( [ . ] [ 0-9 ] +| [ 0-9 ] + ( [ . ] [ 0-9 ] * ) ? ) " ) ; boolean isHTML = idCandidate . matches ( " < . * > " ) ; if ( isAlphaDig || isDotNumber || isDoubleQuoted || isHTML ) { return idCandidate ; } throw new RuntimeException ( "Generated id '" + idCandidate + "'for vertex '" + v + "' is not valid with respect to the . dot language" ) ; } }