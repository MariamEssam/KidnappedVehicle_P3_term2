/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <float.h>
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 50;
	default_random_engine gen;
	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(x, std[0]);

	// This line creates a normal (Gaussian) distribution for y.
	normal_distribution<double> dist_y(y, std[1]);

	// This line creates a normal (Gaussian) distribution for theta.
	normal_distribution<double> dist_theta(theta, std[2]);
	weights.clear();
	for (int i = 0; i < num_particles; i++)
	{
		double sample_x, sample_y, sample_theta;
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);
		Particle p;
		p.id = i + 1;
		p.x = sample_x;
		p.y = sample_y;
		p.theta = sample_theta;
		p.weight = 1;
		weights.push_back(1);
		particles.push_back(p);
	}
	maxweight = 1;
	is_initialized = true;
}
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	for (int i = 0; i < num_particles; i++)
	{
		double temp_x, temp_y, temp_theta;
		if (fabs(yaw_rate) < 0.000001)
		{
			temp_x = particles[i].x + (velocity)*(cos(particles[i].theta))*delta_t;
			temp_y = particles[i].y + (velocity)*(sin(particles[i].theta))*delta_t;
			temp_theta = particles[i].theta;
		}
		else
		{
			temp_x = particles[i].x + (velocity / yaw_rate)*(sin(particles[i].theta + (delta_t*yaw_rate)) - sin(particles[i].theta));
			temp_y = particles[i].y + (velocity / yaw_rate)*(-cos(particles[i].theta + (delta_t*yaw_rate)) + cos(particles[i].theta));
			temp_theta = particles[i].theta + yaw_rate*delta_t;
		}
		// This line creates a normal (Gaussian) distribution for x.
		normal_distribution<double> dist_x(temp_x, std_pos[0]);

		// This line creates a normal (Gaussian) distribution for y.
		normal_distribution<double> dist_y(temp_y, std_pos[1]);

		// This line creates a normal (Gaussian) distribution for theta.
		normal_distribution<double> dist_theta(temp_theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

double ParticleFilter::dataAssociationAndProbCalc(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, double std_landmark[]) {

	double product = 1.0;
	double AssociatedIndex = -1;
	
	for (int i = 0; i < observations.size(); i++)
	{
		double min_distance = DBL_MAX;
		//find nearest landmark
		for (int j = 0; j < predicted.size(); j++)
		{
			double d = CalDistance(predicted[j].x- observations[i].x, predicted[j].y - observations[i].y);
			if (d < min_distance)
			{
				min_distance = d;
				AssociatedIndex = j;
			}
		}
		if (AssociatedIndex != -1)
		{
			product*=MultivariateGaussian(observations[i].x- predicted[AssociatedIndex].x,  observations[i].y- predicted[AssociatedIndex].y, std_landmark[0], std_landmark[1]);
		}
	}

	return product;
}
double ParticleFilter::MultivariateGaussian(double delta_x, double delta_y, double std_x, double std_y)
{
	double normilze_factor = 1.0 / (2.0 * M_PI*std_x*std_y);
	double term1 = pow(delta_x, 2) / (2 * std_x*std_x);
	double term2 = pow(delta_y, 2) / (2 * std_y*std_y);
	return normilze_factor*exp(-(term1 + term2));
}
double ParticleFilter::CalDistance(double delta_x, double delta_y)
{
	return sqrt(pow(delta_x, 2) + pow(delta_y, 2));
}
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	weights.clear();
	maxweight = -1;
	for (int i = 0; i < particles.size(); i++)
	{
		std::vector<LandmarkObs> transformedObservation = std::vector<LandmarkObs>();
		//1-Transformatiom to map coordinate
		double y_p = particles[i].y;
		double x_p = particles[i].x;
		double theta = particles[i].theta;

		for (int i = 0; i < observations.size(); i++)
		{
			LandmarkObs _observation;
			double x_c = observations[i].x;
			double y_c = observations[i].y;
			_observation.x = x_p + (cos(theta)*x_c) - (sin(theta)*y_c);
			_observation.y = y_p + (sin(theta)*x_c) + (cos(theta)*y_c);
			transformedObservation.push_back(_observation);
		}

		//2- Get the in-range landmarks
		vector<LandmarkObs> LandmarksInRange = vector<LandmarkObs>();
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			if(fabs(particles[i].x- map_landmarks.landmark_list[j].x_f)<=sensor_range
			&& fabs(particles[i].y - map_landmarks.landmark_list[j].y_f) <= sensor_range)
			{
				LandmarkObs lobs;
				lobs.id = map_landmarks.landmark_list[j].id_i;
				lobs.x = map_landmarks.landmark_list[j].x_f;
				lobs.y = map_landmarks.landmark_list[j].y_f;
				LandmarksInRange.push_back(lobs);
			}

		}
		particles[i].weight = dataAssociationAndProbCalc(LandmarksInRange, transformedObservation,std_landmark);
		if (maxweight < particles[i].weight)
			maxweight = particles[i].weight;
		weights.push_back(particles[i].weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	vector<Particle> newparticles;
	std::random_device                  rand_dev;
	std::mt19937                        gen(rand_dev());
	//Generate Random Number
	std::uniform_int_distribution<int>  dist(0, num_particles - 1);
	int index = int(dist(gen));
	//Create weight distribution
	uniform_real_distribution<double> d(0.0, maxweight);
	double beta = 0.0;
	for (int i = 0; i < num_particles; i++)
	{
		beta += d(gen) * 2.0* maxweight;
		while (beta>weights[index])
		{
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		newparticles.push_back(particles[index]);
	}

	particles = newparticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	return  particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
