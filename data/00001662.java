public class DuctGraph extends DirectedMultigraph<Duct, DuctGraph.Ordinal> { private static class DuctOrdinalEdgeFactory implements EdgeFactory<Duct, Ordinal> { int count = 0; @Override public DuctGraph.Ordinal createEdge( Duct lhs, Duct rhs ) { return makeOrdinal( 0 ); } public DuctGraph.Ordinal makeOrdinal( int ordinal ) { return new DuctGraph.Ordinal( count++, ordinal ); } } public static class Ordinal { int count; int ordinal; public Ordinal( int count, int ordinal ) { this.count = count; this.ordinal = ordinal; } public int getOrdinal() { return ordinal; } @Override public boolean equals( Object object ) { if( this == object ) return true; Ordinal ordinal = (Ordinal) object; if( count != ordinal.count ) return false; return true; } @Override public int hashCode() { return count; } @Override public String toString() { final StringBuilder sb = new StringBuilder( "Ordinal{" ); sb.append( "count=" ).append( count ); sb.append( ", ordinal=" ).append( ordinal ); sb.append( '}' ); return sb.toString(); } } public DuctGraph() { super( new DuctOrdinalEdgeFactory() ); } public synchronized DuctGraph.Ordinal makeOrdinal( int ordinal ) { return ( (DuctOrdinalEdgeFactory) getEdgeFactory() ).makeOrdinal( ordinal ); } }