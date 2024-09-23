public class BytesComparator implements StreamComparator < BufferedInputStream > , Hasher < byte [ ] > , Comparator < byte [ ] > , Serializable { @ Override public int compare ( byte [ ] lhs , byte [ ] rhs ) { if ( lhs == rhs ) return 0 ; return WritableComparator . compareBytes ( lhs , 0 , lhs . length , rhs , 0 , rhs . length ) ; } @ Override public int compare ( BufferedInputStream lhsStream , BufferedInputStream rhsStream ) { byte [ ] lhs = lhsStream . getBuffer ( ) ; int lhsPos = lhsStream . getPosition ( ) ; int lhsLen = readLen ( lhs , lhsPos ) ; lhsStream . skip ( lhsLen + 4 ) ; byte [ ] rhs = rhsStream . getBuffer ( ) ; int rhsPos = rhsStream . getPosition ( ) ; int rhsLen = readLen ( rhs , rhsPos ) ; rhsStream . skip ( rhsLen + 4 ) ; return WritableComparator . compareBytes ( lhs , lhsPos + 4 , lhsLen , rhs , rhsPos + 4 , rhsLen ) ; } private int readLen ( byte [ ] buffer , int off ) { return ( ( buffer [ off ] & 0xff ) < < 24 ) + ( ( buffer [ off + 1 ] & 0xff ) < < 16 ) + ( ( buffer [ off + 2 ] & 0xff ) < < 8 ) + ( buffer [ off + 3 ] & 0xff ) ; } @ Override public int hashCode ( byte [ ] value ) { return Arrays . hashCode ( value ) ; } }