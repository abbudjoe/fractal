use fractal_core::lifecycle::{Tournament, TournamentConfig};

fn main() -> Result<(), fractal_core::error::FractalError> {
    let tournament = Tournament::new(TournamentConfig::default())?;
    let results = tournament.run_generation()?;

    println!("rank  species                  stability  perplexity  arc_acc  tok/s   fitness");
    for result in results {
        println!(
            "{:<5} {:<24} {:<10.2} {:<11.2} {:<8.2} {:<7.0} {:.2}",
            result.rank,
            result.species,
            result.stability_score,
            result.long_context_perplexity,
            result.arc_accuracy,
            result.tokens_per_sec,
            result.fitness
        );
    }

    Ok(())
}
