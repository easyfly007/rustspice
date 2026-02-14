use crate::schema::Summary;

pub trait SessionApi {
    fn get_summary(&self) -> Summary;
}
